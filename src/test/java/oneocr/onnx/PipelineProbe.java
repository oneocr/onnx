package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Times the pipeline's phases as it is actually structured — the line phase concurrent, the rest
 * sequential — so the next thing worth optimising is chosen from where the time is now rather than
 * from where it used to be. Mirrors {@link OneOcrOnnx#recognize}; if that changes, this must too.
 */
public class PipelineProbe {
    static final int RUNS = 5;

    public static void main(String[] args) throws Exception {
        var env = OrtEnvironment.getEnvironment();
        var paths = ModelPaths.resolve(null);
        System.out.printf("lines=%d detector=%d recognizer=%d classifier=%d%n%n",
                Threads.lines(), Threads.detector(), Threads.recognizer(), Threads.classifier());

        try (var engine = new OneOcrOnnx(paths)) {
            engine.preload(ScriptGroup.LATIN);
            for (var file : args) {
                var image = Raster.of(ImageIO.read(Path.of(file).toFile()));
                System.out.printf("%s  %dx%d%n", Path.of(file).getFileName(), image.width, image.height);
                phases(engine, image);
                System.out.println();
            }
        }
    }

    static void phases(OneOcrOnnx engine, Raster image) throws Exception {
        var whole = best(() -> engine.recognize(image, null, 0));
        var orientation = best(() -> OrientationCorrector.estimate(image, engine.detector, engine.classifier, engine.maxSide));

        var levels = engine.detector.run(image, engine.maxSide);
        var detector = best(() -> engine.detector.run(image, engine.maxSide));
        var vertical = levels.get(0).verticalLayout(engine.scoreThreshold);

        var segmentation = best(() -> shapes(engine, levels, vertical));
        var shapes = shapes(engine, levels, vertical);

        var lineCount = Math.min(engine.maxLines, shapes.size());
        var selected = shapes.subList(0, lineCount);
        var lines = best(() -> Workers.map(lineCount,
                i -> engine.readLine(image, image, selected.get(i), vertical, OcrOptions.defaults(), 0)));

        var linesSerial = withLines(1, () -> best(() -> Workers.map(lineCount,
                i -> engine.readLine(image, image, selected.get(i), vertical, OcrOptions.defaults(), 0))));

        System.out.printf("  %-38s %7.0f ms%n", "whole page, rotation known", whole);
        System.out.printf("  %-38s %7.0f ms   (skipped when rotation is known)%n", "orientation search", orientation);
        System.out.printf("  %-38s %7.0f ms%n", "detector, 1 call + page resize", detector);
        System.out.printf("  %-38s %7.0f ms%n", "segmentation + level merge", segmentation);
        System.out.printf("  %-38s %7.0f ms   (%d lines)%n", "line phase, concurrent", lines, lineCount);
        System.out.printf("  %-38s %7.0f ms   -> line phase scales %.1fx%n",
                "line phase, one thread", linesSerial, linesSerial / lines);
        System.out.printf("  %-38s %7.0f ms%n", "unaccounted", whole - detector - segmentation - lines);

        // The sweep drives its own pools: Workers caches its size at first use, so changing the
        // property alone would vary only the chunking and every row would still run on 16 threads.
        System.out.println("  how the line phase scales, and what is in it:");
        for (var workers : new int[]{1, 2, 4, 8, 16, 24}) {
            var pool = workers == 1 ? null : new java.util.concurrent.ForkJoinPool(workers);
            try {
                var classifyAndRead = best(() -> on(pool, lineCount,
                        i -> engine.readLine(image, image, selected.get(i), vertical, OcrOptions.defaults(), 0)));
                var readOnly = best(() -> on(pool, lineCount,
                        i -> engine.readLine(image, image, selected.get(i), vertical, OcrOptions.defaults().script(ScriptGroup.LATIN), 0)));
                var cropOnly = best(() -> on(pool, lineCount, i -> crop(image, selected.get(i))));
                System.out.printf("    %2d workers   classify+read %7.0f   read only %7.0f   crop only %7.0f ms%n",
                        workers, classifyAndRead, readOnly, cropOnly);
            } finally {
                if (pool != null) pool.shutdown();
            }
        }
    }

    /** Runs a task over {@code [0, count)} on the given pool, or inline when it is null. */
    static Object on(java.util.concurrent.ForkJoinPool pool, int count, Workers.Task<?> task) throws Exception {
        if (pool == null) {
            for (var i = 0; i < count; i++) task.run(i);
            return null;
        }
        var futures = new ArrayList<java.util.concurrent.ForkJoinTask<?>>(count);
        for (var i = 0; i < count; i++) {
            var index = i;
            futures.add(pool.submit(() -> {
                task.run(index);
                return null;
            }));
        }
        for (var f : futures) f.get();
        return null;
    }

    /** The crop half of {@link OneOcrOnnx#readLine}, with no inference, to price the Java side. */
    static Raster crop(Raster upright, LineShape shape) {
        var box = shape.crop();
        return upright.crop(Math.max(0, (int) box.x() - OneOcrOnnx.CROP_PADDING),
                Math.max(0, (int) box.y() - OneOcrOnnx.CROP_PADDING),
                Math.min(upright.width, (int) box.right() + OneOcrOnnx.CROP_PADDING),
                Math.min(upright.height, (int) box.bottom() + OneOcrOnnx.CROP_PADDING));
    }

    static List<LineShape> shapes(OneOcrOnnx engine, List<DetectionMaps> levels, boolean vertical) {
        var perLevel = new ArrayList<List<LineShape>>();
        for (var maps : levels)
            perLevel.add(LineSegmenter.segment(maps, vertical, engine.scoreThreshold, engine.linkThreshold));
        var merged = LevelMerge.merge(perLevel);
        merged.sort(vertical
                ? Comparator.comparingDouble((LineShape s) -> s.crop().x()).reversed()
                : Comparator.comparingDouble(s -> s.crop().y()));
        return merged;
    }

    interface Work {
        Object run() throws Exception;
    }

    static double best(Work work) throws Exception {
        work.run();
        var best = Double.MAX_VALUE;
        for (var i = 0; i < RUNS; i++) {
            var t = System.nanoTime();
            work.run();
            best = Math.min(best, (System.nanoTime() - t) / 1e6);
        }
        return best;
    }

    /** Runs {@code work} with the line pool forced to one thread, then restores the property. */
    static double withLines(int count, Work work) throws Exception {
        var previous = System.getProperty(Threads.LINES);
        System.setProperty(Threads.LINES, String.valueOf(count));
        try {
            return (double) work.run();
        } finally {
            if (previous == null) System.clearProperty(Threads.LINES);
            else System.setProperty(Threads.LINES, previous);
        }
    }
}
