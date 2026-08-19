package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.nio.file.Path;
import java.util.*;

public final class TextDetector implements AutoCloseable {
    public static final int MAX_SIDE = 1536, STRIDE = 4;
    public static final int[] LEVELS = {2, 3, 4};
    static final int MULTIPLE = 32;
    static final String DATA = "data", IM_INFO = "im_info";

    final Model model;

    public TextDetector(OrtEnvironment env, Path file) throws OrtException {
        this(env, file, false);
    }

    public TextDetector(OrtEnvironment env, Path file, boolean gpu) throws OrtException {
        this.model = new Model(env, file, gpu);
    }

    public List<DetectionMaps> run(Raster image, int maxSide) throws OrtException {
        var scale = Math.min((double) maxSide / Math.max(image.width, image.height), 1.0);
        var targetWidth = fit(image.width * scale);
        var targetHeight = fit(image.height * scale);
        var resized = image.resize(targetWidth, targetHeight);

        var data = model.floats(Tensors.chw(resized, false, false), 1, Raster.CHANNELS, targetHeight, targetWidth);
        var info = model.floats(new float[]{targetHeight, targetWidth, 1f}, 1, 3);
        try (data; info) {
            var out = model.run(Map.of(DATA, data, IM_INFO, info), names());
            var levels = new ArrayList<DetectionMaps>(LEVELS.length);
            for (var level : LEVELS)
                levels.add(maps(out, level, targetWidth,
                        (double) targetWidth / image.width, (double) targetHeight / image.height));
            return levels;
        }
    }

    static DetectionMaps maps(Map<String, Tensor> out, int level, int targetWidth, double scaleX, double scaleY) {
        var scores = out.get("scores_hori_fpn" + level);
        var rows = scores.dim(scores.rank() - 2);
        var cols = scores.dim(scores.rank() - 1);
        var built = new DetectionMaps(scores.data, out.get("scores_vert_fpn" + level).data,
                out.get("link_scores_hori_fpn" + level).data, out.get("link_scores_vert_fpn" + level).data,
                out.get("bbox_deltas_hori_fpn" + level).data, out.get("bbox_deltas_vert_fpn" + level).data,
                rows, cols, scaleX, scaleY);
        built.stride = (double) targetWidth / cols;
        return built;
    }

    static String[] names() {
        var names = new ArrayList<String>();
        for (var level : LEVELS) {
            names.add("scores_hori_fpn" + level);
            names.add("scores_vert_fpn" + level);
            names.add("link_scores_hori_fpn" + level);
            names.add("link_scores_vert_fpn" + level);
            names.add("bbox_deltas_hori_fpn" + level);
            names.add("bbox_deltas_vert_fpn" + level);
        }
        return names.toArray(String[]::new);
    }

    static int fit(double value) {
        return Math.max(MULTIPLE, (int) Math.round(value / MULTIPLE) * MULTIPLE);
    }

    @Override
    public void close() {
        model.close();
    }
}
