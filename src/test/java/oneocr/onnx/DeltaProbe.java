package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.*;

public class DeltaProbe {
    public static void main(String[] args) throws Exception {
        var image = Raster.of(ImageIO.read(Path.of(args[0]).toFile()));
        var paths = ModelPaths.resolve(null);
        var env = OrtEnvironment.getEnvironment();
        try (var detector = new TextDetector(env, paths.detector())) {
            var maps = detector.run(image, TextDetector.MAX_SIDE).get(0);
            System.out.printf("image %dx%d  grid %dx%d  scale %.5f,%.5f  stride %d%n",
                    image.width, image.height, maps.rows, maps.cols, maps.scaleX, maps.scaleY, TextDetector.STRIDE);

            var cells = new ArrayList<int[]>();
            for (var r = 0; r < maps.rows; r++)
                for (var c = 0; c < maps.cols; c++)
                    if (maps.score(false, r, c) > 0.5f) cells.add(new int[]{r, c});
            cells.sort(Comparator.comparingDouble((int[] p) -> maps.score(false, p[0], p[1])).reversed());
            System.out.println("active cells: " + cells.size());

            for (var cell : sample(cells)) {
                var r = cell[0];
                var c = cell[1];
                var cx = c * TextDetector.STRIDE + 2.0;
                var cy = r * TextDetector.STRIDE + 2.0;
                var d = new double[8];
                for (var k = 0; k < 8; k++) d[k] = maps.delta(false, k, r, c);
                System.out.printf("cell r=%d c=%d score=%.3f centre(resized)=%.1f,%.1f centre(orig)=%.1f,%.1f%n",
                        r, c, maps.score(false, r, c), cx, cy, cx / maps.scaleX, cy / maps.scaleY);
                System.out.printf("   raw          = %s%n", fmt(d));
                System.out.printf("   centre+raw   = %s%n", fmt(scale(shift(d, cx, cy), maps.scaleX, maps.scaleY)));
                System.out.printf("   centre+raw*4 = %s%n", fmt(scale(shift(times(d, 4), cx, cy), maps.scaleX, maps.scaleY)));
            }
        }
    }

    static List<int[]> sample(List<int[]> cells) {
        var out = new ArrayList<int[]>();
        for (var i = 0; i < cells.size() && out.size() < 5; i += Math.max(1, cells.size() / 5)) out.add(cells.get(i));
        return out;
    }

    static double[] times(double[] d, double k) {
        var out = new double[d.length];
        for (var i = 0; i < d.length; i++) out[i] = d[i] * k;
        return out;
    }

    static double[] shift(double[] d, double cx, double cy) {
        var out = new double[8];
        for (var i = 0; i < 4; i++) {
            out[i * 2] = cx + d[i * 2];
            out[i * 2 + 1] = cy + d[i * 2 + 1];
        }
        return out;
    }

    static double[] scale(double[] d, double sx, double sy) {
        var out = new double[8];
        for (var i = 0; i < 4; i++) {
            out[i * 2] = d[i * 2] / sx;
            out[i * 2 + 1] = d[i * 2 + 1] / sy;
        }
        return out;
    }

    static String fmt(double[] d) {
        var sb = new StringBuilder();
        for (var i = 0; i < d.length; i += 2) sb.append(String.format("(%.1f,%.1f) ", d[i], d[i + 1]));
        return sb.toString();
    }
}
