package oneocr.onnx;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.concurrent.Callable;

@Command(name = "oneocr-onnx", mixinStandardHelpOptions = true, version = "2.0",
        description = "Runs the OneOCR models through ONNX Runtime. No oneocr.dll, works on any platform.")
public final class OnnxOcrMain implements Callable<Integer> {

    @Parameters(index = "0", description = "Image to read.")
    Path image;

    @Option(names = {"-m", "--models"}, description = "Model folder. Default: ~/oneocr/models")
    Path models;

    @Option(names = {"-l", "--lang"}, description = "Force a script instead of detecting it: "
            + "cjk, cyrillic, latin, arabic, devanagari, greek, thai, hebrew, tamil.")
    String language;

    @Option(names = {"-r", "--rotation"}, description = "Force 0, 90, 180 or 270 instead of estimating it. "
            + "A page rendered from a PDF is upright: pass 0 and skip the search, which is the single "
            + "largest saving available and costs nothing in accuracy.")
    Integer rotation;

    @Option(names = "--orientation-search", description = "How hard to look for the rotation when it is "
            + "not given: FULL tries all four (default), EARLY_EXIT stops at upright when it looks "
            + "upright and costs one detector call instead of four. EARLY_EXIT can change which angle "
            + "is chosen; prefer --rotation when the angle is actually known.")
    OrientationSearch orientationSearch = OrientationSearch.FULL;

    @Option(names = "--candidates", split = ",", description = "Confine the script classifier to these "
            + "scripts, keeping it per line: e.g. latin,cjk. Use when a document is mixed but not "
            + "arbitrary. The raw classifier label is noisy — it routes plain English to the cyrillic "
            + "recognizer — and this corrects the routing without forcing one script on every line.")
    java.util.List<String> candidates;

    @Option(names = {"-d", "--detail"}, description = "Also print per-line boxes, script and confidence to stderr.")
    boolean detail;

    @Option(names = "--repeat", description = "Recognise the image N times in one process and report warm timings.")
    int repeat = 1;

    @Option(names = "--gpu", description = "Ask ONNX Runtime for a GPU execution provider if one is available.")
    boolean gpu;

    @Override
    public Integer call() throws Exception {
        var options = options();
        var source = ImageIO.read(image.toFile());
        if (source == null) throw new IllegalArgumentException("cannot read image: " + image);

        var started = System.nanoTime();
        try (var engine = new OneOcrOnnx(ModelPaths.resolve(models), gpu)) {
            var raster = Raster.of(source);
            var result = engine.recognize(raster, options);
            System.out.println(result.fullText());
            report(result, started);
            if (repeat > 1) warm(engine, raster, options);
        }
        return 0;
    }

    OcrOptions options() {
        var options = OcrOptions.defaults().rotation(rotation).orientationSearch(orientationSearch);
        if (language != null) options = options.script(named(language));
        if (candidates != null && !candidates.isEmpty())
            options = options.candidates(candidates.stream().map(OnnxOcrMain::named).toArray(ScriptGroup[]::new));
        return options;
    }

    static ScriptGroup named(String name) {
        return ScriptGroup.byName(name)
                .orElseThrow(() -> new IllegalArgumentException("unknown script: " + name));
    }

    void warm(OneOcrOnnx engine, Raster raster, OcrOptions options) throws Exception {
        var times = new long[repeat];
        for (var i = 0; i < repeat; i++) {
            var t = System.nanoTime();
            engine.recognize(raster, options);
            times[i] = System.nanoTime() - t;
        }
        java.util.Arrays.sort(times);
        System.err.printf("warm over %d runs: median %.0f ms, best %.0f ms%n",
                repeat, times[repeat / 2] / 1e6, times[0] / 1e6);
    }

    void report(OcrResult result, long started) {
        System.err.printf("%d lines, angle %d, %.1f s%n",
                result.getLines().size(), result.getImageAngle(), (System.nanoTime() - started) / 1e9);
        if (!detail) return;
        for (var line : result.getLines())
            System.err.printf("  [%s] %.0f,%.0f %.0fx%.0f  %s%n", line.getScript(), line.getBounds().left(),
                    line.getBounds().top(), line.getBounds().width(), line.getBounds().height(), line.getText());
    }

    public static void main(String[] args) {
        System.setOut(new java.io.PrintStream(new java.io.FileOutputStream(java.io.FileDescriptor.out),
                true, java.nio.charset.StandardCharsets.UTF_8));
        System.exit(new CommandLine(new OnnxOcrMain()).execute(args));
    }
}
