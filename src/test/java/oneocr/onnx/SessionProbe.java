package oneocr.onnx;

import ai.onnxruntime.*;

import javax.imageio.ImageIO;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;

/**
 * Second measurement pass: session sharing, ORT spin-wait, the pure-Java image cost,
 * and the output-copy cost that the CJK vocabulary of 32632 makes large.
 */
public class SessionProbe {
    static OrtEnvironment env = OrtEnvironment.getEnvironment();
    static ModelPaths paths;
    static List<Raster> crops = new ArrayList<>();
    static Raster page;
    static final int WORKERS = 16;

    public static void main(String[] args) throws Exception {
        paths = ModelPaths.resolve(null);
        page = Raster.of(ImageIO.read(Path.of(args[0]).toFile()));
        prepareCrops();
        System.out.printf("page %dx%d, %d crops%n%n", page.width, page.height, crops.size());

        System.out.println("--- pure java, no ONNX at all ---");
        time("Raster.of (getRGB path)", 5, () -> Raster.of(ImageIO.read(Path.of(args[0]).toFile())));
        time("full-page rotateCcw 90", 5, () -> page.rotateCcw(90));
        time("full-page resize to detector size", 5, () -> {
            var s = Math.min((double) TextDetector.MAX_SIDE / Math.max(page.width, page.height), 1.0);
            return page.resize(TextDetector.fit(page.width * s), TextDetector.fit(page.height * s));
        });
        time("resize 99 crops to h=60, sequential", 5, () -> {
            for (var c : crops) recogResize(c);
            return null;
        });
        parallelResize();
        time("chw of 99 resized crops, sequential", 5, () -> {
            for (var c : crops) Tensors.chw(recogResize(c), true, true);
            return null;
        });

        System.out.println();
        System.out.println("--- recognizer session strategy, " + WORKERS + " workers, latin ---");
        recognizer("shared session, intraOp 1", false, false);
        recognizer("shared session, intraOp 1, no spin", false, true);
        recognizer("session per worker, intraOp 1", true, false);
        recognizer("session per worker, intraOp 1, no spin", true, true);

        System.out.println();
        System.out.println("--- output handling: copy whole tensor vs argmax off the buffer ---");
        outputCost(ScriptGroup.LATIN);
        outputCost(ScriptGroup.CJK);

        System.out.println();
        System.out.println("--- detector: all 19 outputs vs the 18 we use ---");
        detectorOutputs();

        System.out.println();
        System.out.println("--- orientation: rotate-then-resize (today) vs resize-then-rotate ---");
        orientation();
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

    static Raster recogResize(Raster crop) {
        var w = Math.max(TextRecognizer.MIN_WIDTH,
                (int) Math.round(TextRecognizer.HEIGHT * (double) crop.width / crop.height));
        return crop.resize(w, TextRecognizer.HEIGHT);
    }

    static void parallelResize() throws Exception {
        var pool = Executors.newFixedThreadPool(WORKERS);
        try {
            time("resize 99 crops to h=60, " + WORKERS + " workers", 5, () -> {
                var futures = new ArrayList<Future<?>>();
                for (var c : crops) futures.add(pool.submit(() -> recogResize(c)));
                for (var f : futures) f.get();
                return null;
            });
        } finally {
            pool.shutdown();
        }
    }

    static OrtSession.SessionOptions options(int intraOp, boolean noSpin) throws OrtException {
        var o = new OrtSession.SessionOptions();
        o.setIntraOpNumThreads(intraOp);
        if (noSpin) o.addConfigEntry("session.intra_op.allow_spinning", "0");
        return o;
    }

    static void recognizer(String label, boolean perWorker, boolean noSpin) throws Exception {
        var sessions = new ArrayList<OrtSession>();
        var count = perWorker ? WORKERS : 1;
        for (var i = 0; i < count; i++)
            sessions.add(env.createSession(paths.recognizer(ScriptGroup.LATIN).toString(), options(1, noSpin)));
        var pool = Executors.newFixedThreadPool(WORKERS);
        var next = new java.util.concurrent.atomic.AtomicInteger();
        try {
            var best = Double.MAX_VALUE;
            for (var run = 0; run < 4; run++) {
                var t = System.nanoTime();
                var futures = new ArrayList<Future<?>>();
                for (var c : crops) {
                    var session = sessions.get(perWorker ? next.getAndIncrement() % count : 0);
                    futures.add(pool.submit(() -> {
                        recognize(session, c);
                        return null;
                    }));
                }
                for (var f : futures) f.get();
                if (run > 0) best = Math.min(best, (System.nanoTime() - t) / 1e6);
            }
            System.out.printf("  %-44s %8.0f ms%n", label, best);
        } finally {
            pool.shutdown();
            for (var s : sessions) s.close();
        }
    }

    static void outputCost(ScriptGroup script) throws Exception {
        try (var session = env.createSession(paths.recognizer(script).toString(), options(1, false))) {
            var crop = crops.get(crops.size() / 2);
            var width = Math.max(TextRecognizer.MIN_WIDTH,
                    (int) Math.round(TextRecognizer.HEIGHT * (double) crop.width / crop.height));
            var chw = Tensors.chw(crop.resize(width, TextRecognizer.HEIGHT), true, true);
            var steps = width / TextRecognizer.STEP_DIVISOR;
            System.out.printf("  %s: width %d -> %d steps x %d vocab = %,d floats per line%n",
                    script.fileName(), width, steps, script.vocabSize, steps * script.vocabSize);
            time("  x99 copy whole tensor to float[] (today)", 3, () -> {
                for (var i = 0; i < 99; i++) run(session, chw, width, steps, true);
                return null;
            });
            time("  x99 argmax straight off the buffer", 3, () -> {
                for (var i = 0; i < 99; i++) run(session, chw, width, steps, false);
                return null;
            });
        }
    }

    static Object run(OrtSession session, float[] chw, int width, int steps, boolean copy) throws Exception {
        var data = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw),
                new long[]{1, Raster.CHANNELS, TextRecognizer.HEIGHT, width});
        var lengths = OnnxTensor.createTensor(env, IntBuffer.wrap(new int[]{steps}), new long[]{1});
        try (data; lengths; var result = session.run(Map.of("data", data, "seq_lengths", lengths))) {
            var tensor = (OnnxTensor) result.get(0);
            var buffer = tensor.getFloatBuffer();
            if (copy) {
                var sink = new float[buffer.remaining()];
                buffer.get(sink);
                return sink;
            }
            var vocab = (int) tensor.getInfo().getShape()[2];
            var out = new int[buffer.remaining() / vocab];
            for (var t = 0; t < out.length; t++) {
                var base = t * vocab;
                var best = 0;
                var bestValue = buffer.get(base);
                for (var v = 1; v < vocab; v++) {
                    var value = buffer.get(base + v);
                    if (value > bestValue) {
                        bestValue = value;
                        best = v;
                    }
                }
                out[t] = best;
            }
            return out;
        }
    }

    static void detectorOutputs() throws Exception {
        try (var session = env.createSession(paths.detector().toString(), options(8, false))) {
            var scale = Math.min((double) TextDetector.MAX_SIDE / Math.max(page.width, page.height), 1.0);
            var w = TextDetector.fit(page.width * scale);
            var h = TextDetector.fit(page.height * scale);
            var chw = Tensors.chw(page.resize(w, h), false, false);
            var wanted = new HashSet<>(Arrays.asList(TextDetector.names()));
            time("all outputs (session.run(inputs))", 5, () -> detect(session, chw, w, h, null));
            time("only the 18 named ones", 5, () -> detect(session, chw, w, h, wanted));
        }
    }

    static Object detect(OrtSession session, float[] chw, int w, int h, Set<String> wanted) throws Exception {
        var data = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, Raster.CHANNELS, h, w});
        var info = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{h, w, 1f}), new long[]{1, 3});
        try (data; info) {
            var inputs = Map.of("data", data, "im_info", info);
            try (var result = wanted == null ? session.run(inputs) : session.run(inputs, wanted)) {
                return result.get(0);
            }
        }
    }

    static void orientation() throws Exception {
        var scale = Math.min((double) TextDetector.MAX_SIDE / Math.max(page.width, page.height), 1.0);
        time("4x rotate full page, then resize each", 3, () -> {
            for (var angle : OrientationCorrector.ANGLES) {
                var r = page.rotateCcw(angle);
                var s = Math.min((double) TextDetector.MAX_SIDE / Math.max(r.width, r.height), 1.0);
                r.resize(TextDetector.fit(r.width * s), TextDetector.fit(r.height * s));
            }
            return null;
        });
        time("resize page once, rotate the small one 4x", 3, () -> {
            var small = page.resize(TextDetector.fit(page.width * scale), TextDetector.fit(page.height * scale));
            for (var angle : OrientationCorrector.ANGLES) small.rotateCcw(angle);
            return null;
        });
    }

    interface Work<T> {
        T run() throws Exception;
    }

    static <T> void time(String label, int runs, Work<T> work) throws Exception {
        work.run();
        var best = Double.MAX_VALUE;
        for (var i = 0; i < runs; i++) {
            var t = System.nanoTime();
            work.run();
            best = Math.min(best, (System.nanoTime() - t) / 1e6);
        }
        System.out.printf("  %-44s %8.1f ms%n", label, best);
    }

    static void recognize(OrtSession session, Raster crop) {
        try {
            var width = Math.max(TextRecognizer.MIN_WIDTH,
                    (int) Math.round(TextRecognizer.HEIGHT * (double) crop.width / crop.height));
            var chw = Tensors.chw(crop.resize(width, TextRecognizer.HEIGHT), true, true);
            run(session, chw, width, width / TextRecognizer.STEP_DIVISOR, true);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
