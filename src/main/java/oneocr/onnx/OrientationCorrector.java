package oneocr.onnx;

import ai.onnxruntime.OrtException;

import java.util.*;

import static oneocr.onnx.TextDetector.STRIDE;

public final class OrientationCorrector {
    public static final int[] ANGLES = {0, 90, 180, 270};
    static final int SAMPLES = 3, HALF_WIDTH = 60, HALF_HEIGHT = 20;
    static final float SCORE_THRESHOLD = 0.5f;
    static final double NO_FLIP = -300.0, VALID_SCRIPT_WEIGHT = 1000.0;

    private OrientationCorrector() {}

    public static int estimate(Raster image, TextDetector detector, ScriptClassifier classifier, int maxSide)
            throws OrtException {
        var bestAngle = 0;
        var bestQuality = -Double.MAX_VALUE;
        for (var angle : ANGLES) {
            var candidate = image.rotateCcw(angle);
            var quality = quality(candidate, detector, classifier, maxSide);
            if (quality > bestQuality) {
                bestQuality = quality;
                bestAngle = angle;
            }
        }
        return bestAngle;
    }

    static double quality(Raster candidate, TextDetector detector, ScriptClassifier classifier, int maxSide)
            throws OrtException {
        var maps = detector.run(candidate, maxSide).get(0);
        var strongest = strongest(maps);
        if (strongest.isEmpty()) return -Double.MAX_VALUE;

        var flips = 0.0;
        var validScripts = 0;
        for (var cell : strongest) {
            var verdict = classifier.classify(cropAround(candidate, maps, cell));
            flips += verdict.flipScore();
            if (verdict.scriptIndex() != 0) validScripts++;
        }
        return validScripts * VALID_SCRIPT_WEIGHT + (strongest.isEmpty() ? NO_FLIP : flips / strongest.size());
    }

    static List<int[]> strongest(DetectionMaps maps) {
        var active = new ArrayList<int[]>();
        for (var r = 0; r < maps.rows; r++)
            for (var c = 0; c < maps.cols; c++)
                if (maps.score(false, r, c) > SCORE_THRESHOLD) active.add(new int[]{r, c});
        active.sort(Comparator.comparingDouble((int[] p) -> maps.score(false, p[0], p[1])).reversed());
        return active.subList(0, Math.min(SAMPLES, active.size()));
    }

    static Raster cropAround(Raster image, DetectionMaps maps, int[] cell) {
        var cx = cell[1] * STRIDE + 2.0;
        var cy = cell[0] * STRIDE + 2.0;
        return image.crop((int) ((cx - HALF_WIDTH) / maps.scaleX), (int) ((cy - HALF_HEIGHT) / maps.scaleY),
                (int) ((cx + HALF_WIDTH) / maps.scaleX), (int) ((cy + HALF_HEIGHT) / maps.scaleY));
    }
}
