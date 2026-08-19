package oneocr.onnx;

import java.nio.file.*;

public final class ModelPaths {
    final Path root;

    public ModelPaths(Path root) {
        this.root = root;
    }

    public static Path defaultRoot() {
        return Path.of(System.getProperty("user.home"), "oneocr", "models");
    }

    public static ModelPaths resolve(Path override) {
        var root = override != null ? override : defaultRoot();
        var paths = new ModelPaths(root);
        if (!Files.exists(paths.detector()))
            throw new IllegalStateException("No OneOCR models under " + root
                    + ". Run oneocr-modelex on Windows once to produce them.");
        return paths;
    }

    public Path detector() {
        return root.resolve("detector").resolve("text_detector.onnx");
    }

    public Path classifier() {
        return root.resolve("classifier").resolve("script_classifier.onnx");
    }

    public Path recognizer(ScriptGroup script) {
        return root.resolve("recognizers").resolve("recognizer_" + script.fileName() + ".onnx");
    }

    public Path vocab(ScriptGroup script) {
        return root.resolve("vocab").resolve("vocab_" + script.fileName() + ".txt");
    }

    public boolean has(ScriptGroup script) {
        return Files.exists(recognizer(script)) && Files.exists(vocab(script));
    }
}
