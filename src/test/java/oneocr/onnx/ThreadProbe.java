package oneocr.onnx;

import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Measures where the ONNX pipeline's time really goes on CPU, and what threading does about it.
 * The recognizer's batch dimension is fixed at 1 in the model, so batching is not available;
 * this probe exists to find out what concurrency over independent lines is worth instead.
 */
public class ThreadProbe {
    static OrtEnvironment env;
    static ModelPaths paths;
    static List<Raster> crops = new ArrayList<>();
    static Raster page;

    public static void main(String[] args) throws Exception {
        env = OrtEnvironment.getEnvironment();
        paths = ModelPaths.resolve(null);
        page = Raster.of(ImageIO.read(Path.of(args[0]).toFile()));
        System.out.printf("page %dx%d, cores %d%n", page.width, page.height, Runtime.getRuntime().availableProcessors());

        prepareCrops();
        System.out.printf("%d line crops prepared%n%n", crops.size());

        System.out.println("--- recognizer, 99 lines, latin ---");
        for (var intra : new int[]{0, 1, 2, 4, 16})
            for (var workers : new int[]{1, 4, 8, 16})
                if (!(intra == 0 && workers > 1)) recognizerRun(intra, workers);

        System.out.println();
        System.out.println("--- classifier, 99 crops ---");
        for (var intra : new int[]{0, 1})
            for (var workers : new int[]{1, 8, 16})
                if (!(intra == 0 && workers > 1)) classifierRun(intra, workers);

        System.out.println();
        System.out.println("--- detector, one full page call ---");
        for (var intra : new int[]{0, 1, 4, 8, 16}) detectorRun(intra);
    }

    static void prepareCrops() throws Exception {
        try (var detector = new TextDetector(env, paths.detector(), false)) {
            var levels = detector.run(page, TextDetector.MAX_SIDE);
            var vertical = levels.get(0).verticalLayout(0.5f);
            var perLevel = new ArrayList<List<LineShape>>();
            for (var maps : levels) perLevel.add(LineSegmenter.segment(maps, vertical, 0.5f, 0f));
            for (var shape : LevelMerge.merge(perLevel)) {
                var b = shape.crop();
                crops.add(page.crop(Math.max(0, (int) b.x() - 2), Math.max(0, (int) b.y() - 2),
                        Math.min(page.width, (int) b.right() + 2), Math.min(page.height, (int) b.bottom() + 2)));
            }
        }
    }

    static OrtSession.SessionOptions options(int intraOp) throws OrtException {
        var o = new OrtSession.SessionOptions();
        if (intraOp > 0) o.setIntraOpNumThreads(intraOp);
        return o;
    }

    static void recognizerRun(int intraOp, int workers) throws Exception {
        try (var session = env.createSession(paths.recognizer(ScriptGroup.LATIN).toString(), options(intraOp))) {
            var vocab = Vocabulary.load(paths.vocab(ScriptGroup.LATIN), ScriptGroup.LATIN.vocabSize);
            report("recognizer", intraOp, workers, sweep(workers, crops, c -> recognize(session, c, vocab)));
        }
    }

    static void classifierRun(int intraOp, int workers) throws Exception {
        try (var session = env.createSession(paths.classifier().toString(), options(intraOp))) {
            report("classifier", intraOp, workers, sweep(workers, crops, c -> classify(session, c)));
        }
    }

    static void detectorRun(int intraOp) throws Exception {
        try (var session = env.createSession(paths.detector().toString(), options(intraOp))) {
            var one = List.of(page);
            report("detector", intraOp, 1, sweep(1, one, p -> detect(session, p)));
        }
    }

    interface Job { void apply(Raster r) throws Exception; }

    /** Runs the job over every item, warm, three times; returns the best wall clock in ms. */
    static double sweep(int workers, List<Raster> items, Job job) throws Exception {
        for (var item : items.subList(0, Math.min(4, items.size()))) job.apply(item);
        var best = Double.MAX_VALUE;
        var pool = workers == 1 ? null : Executors.newFixedThreadPool(workers);
        try {
            for (var run = 0; run < 3; run++) {
                var t = System.nanoTime();
                if (pool == null) {
                    for (var item : items) job.apply(item);
                } else {
                    var futures = new ArrayList<Future<?>>();
                    for (var item : items) futures.add(pool.submit(() -> { job.apply(item); return null; }));
                    for (var f : futures) f.get();
                }
                best = Math.min(best, (System.nanoTime() - t) / 1e6);
            }
        } finally {
            if (pool != null) pool.shutdown();
        }
        return best;
    }

    static void report(String stage, int intraOp, int workers, double ms) {
        System.out.printf("  intraOp %-2s  workers %-3d  %8.0f ms%n",
                intraOp == 0 ? "def" : String.valueOf(intraOp), workers, ms);
    }

    static void recognize(OrtSession session, Raster crop, Vocabulary vocab) throws Exception {
        var width = Math.max(TextRecognizer.MIN_WIDTH,
                (int) Math.round(TextRecognizer.HEIGHT * (double) crop.width / crop.height));
        var resized = crop.resize(width, TextRecognizer.HEIGHT);
        var data = OnnxTensor.createTensor(env, FloatBuffer.wrap(Tensors.chw(resized, true, true)),
                new long[]{1, Raster.CHANNELS, TextRecognizer.HEIGHT, width});
        var lengths = OnnxTensor.createTensor(env,
                IntBuffer.wrap(new int[]{width / TextRecognizer.STEP_DIVISOR}), new long[]{1});
        try (data; lengths; var result = session.run(Map.of("data", data, "seq_lengths", lengths))) {
            var buffer = ((OnnxTensor) result.get(0)).getFloatBuffer();
            var sink = new float[buffer.remaining()];
            buffer.get(sink);
        }
    }

    static void classify(OrtSession session, Raster crop) throws Exception {
        var resized = crop.resize(ScriptClassifier.WIDTH, ScriptClassifier.HEIGHT);
        var data = OnnxTensor.createTensor(env, FloatBuffer.wrap(Tensors.chw(resized, true, true)),
                new long[]{1, Raster.CHANNELS, ScriptClassifier.HEIGHT, ScriptClassifier.WIDTH});
        try (data; var result = session.run(Map.of("data", data))) {
            result.get("script_id_score").orElseThrow();
        }
    }

    static void detect(OrtSession session, Raster image) throws Exception {
        var scale = Math.min((double) TextDetector.MAX_SIDE / Math.max(image.width, image.height), 1.0);
        var w = TextDetector.fit(image.width * scale);
        var h = TextDetector.fit(image.height * scale);
        var resized = image.resize(w, h);
        var data = OnnxTensor.createTensor(env, FloatBuffer.wrap(Tensors.chw(resized, false, false)),
                new long[]{1, Raster.CHANNELS, h, w});
        var info = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{h, w, 1f}), new long[]{1, 3});
        try (data; info; var result = session.run(Map.of("data", data, "im_info", info))) {
            result.get("scores_hori_fpn2").orElseThrow();
        }
    }
}
