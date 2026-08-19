package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;

import javax.imageio.ImageIO;
import java.nio.file.Path;

public class PhaseProbe {
    public static void main(String[] args) throws Exception {
        var image = Raster.of(ImageIO.read(Path.of(args[0]).toFile()));
        var paths = ModelPaths.resolve(null);
        var env = OrtEnvironment.getEnvironment();

        var gpu = Boolean.getBoolean("oneocr.gpu");
        System.out.println("provider: " + (gpu ? "CUDA" : "CPU"));
        try (var detector = new TextDetector(env, paths.detector(), gpu);
             var classifier = new ScriptClassifier(env, paths.classifier(), gpu);
             var recognizer = new TextRecognizer(env, paths, gpu)) {

            detector.run(image, TextDetector.MAX_SIDE);
            var levels = timed("detector (1 call, 3 FPN levels)", 5, () -> detector.run(image, TextDetector.MAX_SIDE));

            var vertical = levels.get(0).verticalLayout(0.5f);
            var shapes = new java.util.ArrayList<LineShape>();
            for (var maps : levels) shapes.addAll(LineSegmenter.segment(maps, vertical, 0.5f, 0f));
            var merged = LevelMerge.merge(shapes.stream().map(java.util.List::of).toList());
            System.out.printf("lines found: %d%n", merged.size());

            timed("segmentation + merge (pure java)", 5, () -> {
                var s = new java.util.ArrayList<LineShape>();
                for (var maps : levels) s.addAll(LineSegmenter.segment(maps, vertical, 0.5f, 0f));
                return LevelMerge.merge(s.stream().map(java.util.List::of).toList());
            });

            var crops = new java.util.ArrayList<Raster>();
            for (var shape : merged) {
                var b = shape.crop();
                crops.add(image.crop(Math.max(0, (int) b.x() - 2), Math.max(0, (int) b.y() - 2),
                        Math.min(image.width, (int) b.right() + 2), Math.min(image.height, (int) b.bottom() + 2)));
            }

            timed("resample " + crops.size() + " crops to h=60 (pure java, no ONNX)", 5, () -> {
                for (var crop : crops) crop.resize(Math.max(16, (int) Math.round(60.0 * crop.width / crop.height)), 60);
                return null;
            });

            timed("classifier x" + crops.size() + " (ONNX)", 3, () -> {
                for (var crop : crops) classifier.classify(crop);
                return null;
            });

            timed("recognizer x" + crops.size() + " dynamic widths (ONNX)", 3, () -> {
                for (var crop : crops) recognizer.recognize(crop, ScriptGroup.LATIN);
                return null;
            });

            var fixed = crops.get(crops.size() / 2);
            var n = crops.size();
            timed("recognizer x" + n + " SAME crop, constant shape (ONNX)", 3, () -> {
                for (var i = 0; i < n; i++) recognizer.recognize(fixed, ScriptGroup.LATIN);
                return null;
            });
        }
    }

    interface Work<T> { T run() throws Exception; }

    static <T> T timed(String label, int runs, Work<T> work) throws Exception {
        T last = null;
        var best = Long.MAX_VALUE;
        for (var i = 0; i < runs; i++) {
            var t = System.nanoTime();
            last = work.run();
            best = Math.min(best, System.nanoTime() - t);
        }
        System.out.printf("%-48s %6.0f ms%n", label, best / 1e6);
        return last;
    }
}
