package oneocr.onnx;

public final class Resampler {
    static final double SUPPORT = 2.0, A = -0.5;

    private Resampler() {}

    static double bicubic(double x) {
        if (x < 0) x = -x;
        if (x < 1.0) return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
        return 0.0;
    }

    public static Raster resize(Raster src, int width, int height) {
        if (src.width == width && src.height == height) return src;
        var horizontal = pass(src.px, src.width, src.height, width, true);
        var vertical = pass(horizontal, width, src.height, height, false);
        return new Raster(width, height, vertical);
    }

    static float[] pass(float[] in, int inWidth, int inHeight, int outSize, boolean horizontal) {
        var inSize = horizontal ? inWidth : inHeight;
        if (inSize == outSize) return in;
        var coeffs = coefficients(inSize, outSize);
        var outWidth = horizontal ? outSize : inWidth;
        var outHeight = horizontal ? inHeight : outSize;
        var out = new float[outWidth * outHeight * Raster.CHANNELS];
        var rows = horizontal ? inHeight : inWidth;

        for (var line = 0; line < rows; line++) {
            for (var i = 0; i < outSize; i++) {
                var bounds = coeffs.bounds()[i];
                var kernel = coeffs.kernels()[i];
                for (var c = 0; c < Raster.CHANNELS; c++) {
                    var sum = 0.0;
                    for (var k = 0; k < kernel.length; k++) {
                        var index = horizontal
                                ? (line * inWidth + bounds + k) * Raster.CHANNELS + c
                                : ((bounds + k) * inWidth + line) * Raster.CHANNELS + c;
                        sum += kernel[k] * in[index];
                    }
                    var target = horizontal
                            ? (line * outWidth + i) * Raster.CHANNELS + c
                            : (i * outWidth + line) * Raster.CHANNELS + c;
                    out[target] = clamp(sum);
                }
            }
        }
        return out;
    }

    static float clamp(double v) {
        return v < 0 ? 0 : v > 255 ? 255 : (float) v;
    }

    static Coefficients coefficients(int inSize, int outSize) {
        var scale = (double) inSize / outSize;
        var filterScale = Math.max(scale, 1.0);
        var support = SUPPORT * filterScale;
        var bounds = new int[outSize];
        var kernels = new double[outSize][];

        for (var xx = 0; xx < outSize; xx++) {
            var center = (xx + 0.5) * scale;
            var step = 1.0 / filterScale;
            var min = Math.max((int) (center - support + 0.5), 0);
            var max = Math.min((int) (center + support + 0.5), inSize);
            var count = Math.max(max - min, 1);
            var kernel = new double[count];
            var total = 0.0;
            for (var x = 0; x < count; x++) {
                var w = bicubic((x + min - center + 0.5) * step);
                kernel[x] = w;
                total += w;
            }
            if (total != 0.0) for (var x = 0; x < count; x++) kernel[x] /= total;
            bounds[xx] = min;
            kernels[xx] = kernel;
        }
        return new Coefficients(bounds, kernels);
    }
}
