package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;

import javax.imageio.ImageIO;
import java.nio.file.Path;
import java.util.*;

public class FpnProbe {
    static final int[] LEVELS = {2, 3, 4};

    public static void main(String[] args) throws Exception {
        var image = Raster.of(ImageIO.read(Path.of(args[0]).toFile()));
        var paths = ModelPaths.resolve(null);
        var env = OrtEnvironment.getEnvironment();

        var scale = Math.min((double) TextDetector.MAX_SIDE / Math.max(image.width, image.height), 1.0);
        var tw = TextDetector.fit(image.width * scale);
        var th = TextDetector.fit(image.height * scale);
        var resized = image.resize(tw, th);
        System.out.printf("image %dx%d -> resized %dx%d%n", image.width, image.height, tw, th);

        try (var model = new Model(env, paths.detector())) {
            var data = model.floats(Tensors.chw(resized, false, false), 1, Raster.CHANNELS, th, tw);
            var info = model.floats(new float[]{th, tw, 1f}, 1, 3);
            var names = new ArrayList<String>();
            for (var l : LEVELS) {
                names.add("scores_hori_fpn" + l);
                names.add("bbox_deltas_hori_fpn" + l);
            }
            try (data; info) {
                var out = model.run(Map.of("data", data, "im_info", info), names.toArray(String[]::new));
                for (var l : LEVELS) report(out.get("scores_hori_fpn" + l), out.get("bbox_deltas_hori_fpn" + l), l, tw, th);
            }
        }
    }

    static void report(Tensor scores, Tensor deltas, int level, int tw, int th) {
        var rows = scores.dim(scores.rank() - 2);
        var cols = scores.dim(scores.rank() - 1);
        var stride = (double) tw / cols;
        var cells = new ArrayList<int[]>();
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                if (scores.data[r * cols + c] > 0.5f) cells.add(new int[]{r, c});
        cells.sort(Comparator.comparingDouble((int[] p) -> scores.data[p[0] * cols + p[1]]).reversed());

        System.out.printf("%n=== fpn%d  grid %dx%d  stride %.1f  active %d%n", level, rows, cols, stride, cells.size());
        if (cells.isEmpty()) return;

        var top = 0.0;
        var bottom = 0.0;
        var width = 0.0;
        var n = Math.min(20, cells.size());
        for (var i = 0; i < n; i++) {
            var r = cells.get(i)[0];
            var c = cells.get(i)[1];
            var cy = r * stride + stride / 2;
            var d = new double[8];
            for (var k = 0; k < 8; k++) d[k] = deltas.data[(k * rows + r) * cols + c];
            if (i < 3) System.out.printf("   r=%d c=%d score=%.3f centre=%.1f,%.1f raw=%s%n",
                    r, c, scores.data[r * cols + c], c * stride + stride / 2, cy, DeltaProbe.fmt(d));
            top += d[1];
            bottom += d[5];
            width += d[2] - d[0];
        }
        System.out.printf("   mean over %d cells: dTop=%.3f dBottom=%.3f height=%.3f width=%.3f%n",
                n, top / n, bottom / n, (bottom - top) / n, width / n);
        System.out.printf("   if unit=stride(%.1f): height=%.1f px | if unit=1: height=%.1f px%n",
                stride, (bottom - top) / n * stride, (bottom - top) / n);
    }
}
