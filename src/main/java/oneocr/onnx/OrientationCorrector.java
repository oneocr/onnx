package oneocr.onnx;

import ai.onnxruntime.OrtException;

import java.util.*;

import static oneocr.onnx.TextDetector.STRIDE;

/**
 * Decides which way up a page is, by detecting text at all four quarter turns and asking the script
 * classifier which one reads like writing.
 *
 * <p>Two things make this cheaper than the obvious loop, and neither changes the answer. The page is
 * resized to model resolution ONCE and the small raster is rotated, rather than rotating the full
 * page and resizing each of the four — a quarter turn only swaps the axes, and the resize targets
 * swap with them, so the same pixels arrive at the detector either way. And under
 * {@link OrientationSearch#FULL} the four candidates are independent, so they run concurrently.
 *
 * <p>The full-resolution rotation is still made, because the crops fed to the script classifier are
 * taken from it: sampling them from the downscaled copy would be a real change to what the
 * classifier sees. Rotating a whole page costs a fraction of resizing one.
 *
 * <p>It remains far cheaper still not to search at all. A page rendered from a PDF is already
 * upright, and a caller that knows the angle should pass it. {@link OrientationSearch#EARLY_EXIT} is
 * for a caller who believes the pages are upright but will not vouch for it.
 *
 * <p>Do not try to save time here by lowering {@code maxSide}. It was measured and it fails
 * erratically rather than gracefully — right at 1024, wrong on four images of five at 768, right
 * again at 512 and 384, wrong at 256. The decision is fragile; see {@code 26-prp.01.measurements.md}.
 */
public final class OrientationCorrector {
    public static final int[] ANGLES = {0, 90, 180, 270};
    static final int SAMPLES = 3, HALF_WIDTH = 60, HALF_HEIGHT = 20;
    static final float SCORE_THRESHOLD = 0.5f;
    static final double VALID_SCRIPT_WEIGHT = 1000.0;

    private OrientationCorrector() {}

    public static int estimate(Raster image, TextDetector detector, ScriptClassifier classifier, int maxSide)
            throws OrtException {
        return estimate(image, detector, classifier, maxSide, OrientationSearch.FULL);
    }

    public static int estimate(Raster image, TextDetector detector, ScriptClassifier classifier, int maxSide,
                              OrientationSearch search) throws OrtException {
        var target = TextDetector.targetSize(image.width, image.height, maxSide);
        var small = image.resize(target[0], target[1]);

        var qualities = new Quality[ANGLES.length];
        var from = 0;
        if (search == OrientationSearch.EARLY_EXIT) {
            qualities[0] = quality(image, small, ANGLES[0], detector, classifier);
            if (qualities[0].upright()) return ANGLES[0];
            from = 1;
        }

        var remaining = ANGLES.length - from;
        var offset = from;
        try {
            Workers.split(remaining, i -> {
                try {
                    qualities[offset + i] = quality(image, small, ANGLES[offset + i], detector, classifier);
                } catch (OrtException failed) {
                    throw new Workers.Wrapped(failed);
                }
            });
        } catch (Workers.Wrapped wrapped) {
            throw (OrtException) wrapped.getCause();
        }

        var bestAngle = 0;
        var bestScore = -Double.MAX_VALUE;
        for (var i = 0; i < ANGLES.length; i++) {
            if (qualities[i] == null) continue;
            var score = qualities[i].score();
            if (score > bestScore) {
                bestScore = score;
                bestAngle = ANGLES[i];
            }
        }
        return bestAngle;
    }

    /**
     * How much this rotation looks like text the right way up.
     *
     * @param found         whether any cell scored above the detection threshold at all
     * @param validScripts  how many of the sampled crops the classifier called a real script
     * @param meanFlipScore the classifier's mean flip score over those samples; its SIGN is what
     *                      distinguishes an upright page from the same page upside down, both of
     *                      which usually score full marks on {@code validScripts}
     */
    record Quality(boolean found, int validScripts, double meanFlipScore) {
        double score() {
            return found ? validScripts * VALID_SCRIPT_WEIGHT + meanFlipScore : -Double.MAX_VALUE;
        }

        /** Confident enough that the other three rotations need not be tried. */
        boolean upright() {
            return found && validScripts == SAMPLES && meanFlipScore > 0;
        }
    }

    static Quality quality(Raster image, Raster small, int angle, TextDetector detector, ScriptClassifier classifier)
            throws OrtException {
        var candidate = image.rotateCcw(angle);
        var maps = detector.runResized(small.rotateCcw(angle), candidate.width, candidate.height).get(0);
        var strongest = strongest(maps);
        if (strongest.isEmpty()) return new Quality(false, 0, 0);

        var flips = 0.0;
        var validScripts = 0;
        for (var cell : strongest) {
            var verdict = classifier.classify(cropAround(candidate, maps, cell));
            flips += verdict.flipScore();
            if (verdict.scriptIndex() != 0) validScripts++;
        }
        return new Quality(true, validScripts, flips / strongest.size());
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
