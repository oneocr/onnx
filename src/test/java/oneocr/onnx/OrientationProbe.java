package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.List;

/**
 * Asks whether the orientation search has to run at full detector resolution.
 *
 * <p>It costs four detector calls, and the detector's cost scales with the area it is given, so a
 * smaller {@code maxSide} should cut it quadratically. What it decides is only one of four discrete
 * answers, which does not obviously need 1536 pixels to reach. This prints the answer and the cost
 * at each resolution for every image given, so agreement can be read off directly — the expected
 * angle is taken from the name when it ends in a rotation, otherwise the full-resolution answer is
 * treated as the reference.
 */
public class OrientationProbe {
    static final int[] SIDES = {1536};
    static final int RUNS = 3;

    public static void main(String[] args) throws Exception {
        var env = OrtEnvironment.getEnvironment();
        var paths = ModelPaths.resolve(null);
        var disagreements = 0;

        try (var detector = new TextDetector(env, paths.detector(), false);
             var classifier = new ScriptClassifier(env, paths.classifier(), false)) {
            for (var file : args) {
                var name = Path.of(file).getFileName().toString();
                var image = Raster.of(ImageIO.read(Path.of(file).toFile()));
                var expected = expected(name);
                System.out.printf("%s  %dx%d%s%n", name, image.width, image.height,
                        expected == null ? "" : "  (should be " + expected + ")");

                // Is the right angle separably strong? If the score at the correct rotation is
                // always maximal and a wrong rotation never reaches that mark, the search can stop
                // after the first candidate on an upright page and pay one detector call, not four.
                var fit = TextDetector.targetSize(image.width, image.height, TextDetector.MAX_SIDE);
                var small = image.resize(fit[0], fit[1]);
                System.out.print("  quality per angle: ");
                for (var angle : OrientationCorrector.ANGLES) {
                    var q = OrientationCorrector.quality(image, small, angle, detector, classifier);
                    System.out.printf("%d=%s%s  ", angle, q.found() ? String.format("%.0f", q.score()) : "none",
                            q.upright() ? " UPRIGHT" : "");
                }
                System.out.println();

                Integer reference = null;
                for (var side : SIDES) {
                    var angle = OrientationCorrector.estimate(image, detector, classifier, side);
                    var best = Double.MAX_VALUE;
                    for (var i = 0; i < RUNS; i++) {
                        var t = System.nanoTime();
                        OrientationCorrector.estimate(image, detector, classifier, side);
                        best = Math.min(best, (System.nanoTime() - t) / 1e6);
                    }
                    if (reference == null) reference = angle;
                    var target = expected != null ? expected : reference;
                    var agrees = angle == target;
                    if (!agrees) disagreements++;
                    System.out.printf("  maxSide %-5d  angle %-4d  %7.0f ms   %s%n",
                            side, angle, best, agrees ? "agrees" : "DISAGREES, wanted " + target);
                }
                System.out.println();
            }
        }
        System.out.println(disagreements == 0
                ? "every resolution agreed" : disagreements + " disagreement(s)");
    }

    /** The angle a fixture named "...rot90.jpg" is known to need; null when the name says nothing. */
    static Integer expected(String name) {
        for (var angle : List.of(90, 180, 270)) if (name.contains("rot" + angle)) return angle;
        return name.contains("rot") ? null : 0;
    }
}
