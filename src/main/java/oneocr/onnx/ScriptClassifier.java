package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.nio.file.Path;
import java.util.Map;

public final class ScriptClassifier implements AutoCloseable {
    public static final int WIDTH = 200, HEIGHT = 60;
    static final String DATA = "data", SCRIPT_ID_SCORE = "script_id_score", FLIP_SCORE = "flip_score";

    final Model model;

    public ScriptClassifier(OrtEnvironment env, Path file, boolean gpu) throws OrtException {
        this.model = new Model(env, file, gpu);
    }

    public ScriptVerdict classify(Raster crop) throws OrtException {
        var resized = crop.resize(WIDTH, HEIGHT);
        var data = model.floats(Tensors.chw(resized, true, true), 1, Raster.CHANNELS, HEIGHT, WIDTH);
        try (data) {
            var out = model.run(Map.of(DATA, data), SCRIPT_ID_SCORE, FLIP_SCORE);
            var scores = out.get(SCRIPT_ID_SCORE);
            var flip = out.get(FLIP_SCORE);
            return new ScriptVerdict(Tensors.argmax(scores.data, 0, scores.length()),
                    flip.length() > 0 ? flip.at(0) : 0f);
        }
    }

    @Override
    public void close() {
        model.close();
    }
}
