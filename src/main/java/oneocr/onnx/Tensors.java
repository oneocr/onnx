package oneocr.onnx;

public final class Tensors {
    private Tensors() {}

    public static float[] chw(Raster r, boolean bgr, boolean scaleTo01) {
        var w = r.width;
        var h = r.height;
        var out = new float[Raster.CHANNELS * w * h];
        var divisor = scaleTo01 ? 255f : 1f;
        for (var c = 0; c < Raster.CHANNELS; c++) {
            var source = bgr ? Raster.CHANNELS - 1 - c : c;
            var base = c * w * h;
            for (var y = 0; y < h; y++) {
                var row = y * w;
                for (var x = 0; x < w; x++) out[base + row + x] = r.px[(row + x) * Raster.CHANNELS + source] / divisor;
            }
        }
        return out;
    }

    public static int argmax(float[] values, int from, int count) {
        var best = from;
        for (var i = from + 1; i < from + count; i++) if (values[i] > values[best]) best = i;
        return best - from;
    }
}
