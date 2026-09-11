package oneocr.onnx;

/**
 * Separable bicubic resampling, the same filter and the same coefficients Pillow uses, which is
 * what OneOCR's own preprocessing matches.
 *
 * <p>Both passes accumulate in double over the kernel taps in ascending order and clamp once at the
 * end, exactly as the first version of this class did, so the output is bit-identical. What changed
 * is only the order in which memory is walked and who walks it: the vertical pass used to iterate
 * columns on the outside and stride through the source by a whole row per tap, and neither pass
 * used more than one thread. Full-page resizing is on the critical path of every detector call, and
 * of every candidate in the orientation search.
 */
public final class Resampler {
    static final double SUPPORT = 2.0, A = -0.5;

    /**
     * Output floats above which a pass is split across threads. Full-page resizes are far above it
     * and line crops far below, which is what we want: the page resize is a sequential stage and
     * has the machine to itself, while a line crop is already being resized inside a line worker.
     */
    static final int PARALLEL_FLOOR = 1 << 19;

    private Resampler() {}

    static double bicubic(double x) {
        if (x < 0) x = -x;
        if (x < 1.0) return ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
        if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * A;
        return 0.0;
    }

    public static Raster resize(Raster src, int width, int height) {
        if (src.width == width && src.height == height) return src;
        var horizontal = horizontal(src.px, src.width, src.height, width);
        var vertical = vertical(horizontal, width, src.height, height);
        return new Raster(width, height, vertical);
    }

    /** Resamples along x. Rows are independent; each output row reads only its own source row. */
    static float[] horizontal(float[] in, int inWidth, int inHeight, int outWidth) {
        if (inWidth == outWidth) return in;
        var coeffs = coefficients(inWidth, outWidth);
        var bounds = coeffs.bounds();
        var kernels = coeffs.kernels();
        var out = new float[outWidth * inHeight * Raster.CHANNELS];

        each(inHeight, outWidth * Raster.CHANNELS, row -> {
            var sourceRow = row * inWidth * Raster.CHANNELS;
            var targetRow = row * outWidth * Raster.CHANNELS;
            for (var x = 0; x < outWidth; x++) {
                var kernel = kernels[x];
                var first = sourceRow + bounds[x] * Raster.CHANNELS;
                var red = 0.0;
                var green = 0.0;
                var blue = 0.0;
                for (var k = 0; k < kernel.length; k++) {
                    var weight = kernel[k];
                    var at = first + k * Raster.CHANNELS;
                    red += weight * in[at];
                    green += weight * in[at + 1];
                    blue += weight * in[at + 2];
                }
                var target = targetRow + x * Raster.CHANNELS;
                out[target] = clamp(red);
                out[target + 1] = clamp(green);
                out[target + 2] = clamp(blue);
            }
        });
        return out;
    }

    /**
     * Resamples along y. Each output row is a weighted sum of a few whole source rows, so it is
     * accumulated a source row at a time — every read and every write then runs straight down
     * memory, where the previous version jumped a row per element.
     */
    static float[] vertical(float[] in, int width, int inHeight, int outHeight) {
        if (inHeight == outHeight) return in;
        var coeffs = coefficients(inHeight, outHeight);
        var bounds = coeffs.bounds();
        var kernels = coeffs.kernels();
        var stride = width * Raster.CHANNELS;
        var out = new float[stride * outHeight];

        each(outHeight, stride, row -> {
            var kernel = kernels[row];
            var accumulator = new double[stride];
            for (var k = 0; k < kernel.length; k++) {
                var weight = kernel[k];
                var sourceRow = (bounds[row] + k) * stride;
                for (var i = 0; i < stride; i++) accumulator[i] += weight * in[sourceRow + i];
            }
            var targetRow = row * stride;
            for (var i = 0; i < stride; i++) out[targetRow + i] = clamp(accumulator[i]);
        });
        return out;
    }

    /** Runs {@code body} over every output row, splitting across threads only when it pays. */
    static void each(int rows, int floatsPerRow, java.util.function.IntConsumer body) {
        if ((long) rows * floatsPerRow >= PARALLEL_FLOOR) Workers.split(rows, body);
        else for (var row = 0; row < rows; row++) body.accept(row);
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
