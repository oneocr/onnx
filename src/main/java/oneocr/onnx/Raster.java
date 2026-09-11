package oneocr.onnx;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.SinglePixelPackedSampleModel;

/** An image as planar-free interleaved RGB floats in 0..255, which is what every model here wants. */
public final class Raster {
    public static final int CHANNELS = 3;

    /** Pixels above which a whole-image copy is split across threads. */
    static final int PARALLEL_FLOOR = 1 << 17;

    final int width, height;
    final float[] px;

    Raster(int width, int height, float[] px) {
        this.width = width;
        this.height = height;
        this.px = px;
    }

    public int width() {
        return width;
    }

    public int height() {
        return height;
    }

    /**
     * Reads a {@link BufferedImage}.
     *
     * <p>Where the image's own storage is already interleaved bytes or packed ints in sRGB, the
     * bands are read straight out of the data buffer. {@code getRGB} is the general path and goes
     * through the colour model one pixel at a time; it stays as the fallback, and it is what
     * decides the answer for anything unusual — a grayscale or CMYK image, a non-sRGB colour space,
     * premultiplied alpha, an indexed palette — where a direct read would not give the same
     * numbers. The two paths are asserted equal by {@code RasterProbe}.
     */
    public static Raster of(BufferedImage image) {
        var direct = direct(image);
        return direct != null ? direct : viaColorModel(image);
    }

    static Raster viaColorModel(BufferedImage image) {
        var w = image.getWidth();
        var h = image.getHeight();
        var argb = image.getRGB(0, 0, w, h, null, 0, w);
        var px = new float[w * h * CHANNELS];
        rows(h, w, y -> {
            for (var x = 0; x < w; x++) {
                var i = y * w + x;
                var p = argb[i];
                px[i * CHANNELS] = (p >> 16) & 0xFF;
                px[i * CHANNELS + 1] = (p >> 8) & 0xFF;
                px[i * CHANNELS + 2] = p & 0xFF;
            }
        });
        return new Raster(w, h, px);
    }

    /** Null when this image's storage is not one we can read without the colour model. */
    static Raster direct(BufferedImage image) {
        var model = image.getColorModel();
        if (image.isAlphaPremultiplied() || !model.getColorSpace().isCS_sRGB()) return null;

        var raster = image.getRaster();
        var w = image.getWidth();
        var h = image.getHeight();
        if (raster.getWidth() != w || raster.getHeight() != h) return null;

        var sampleModel = raster.getSampleModel();
        var buffer = raster.getDataBuffer();
        if (buffer.getNumBanks() != 1) return null;
        var origin = buffer.getOffset();
        var px = new float[w * h * CHANNELS];

        if (sampleModel instanceof PixelInterleavedSampleModel interleaved
                && buffer instanceof DataBufferByte bytes && interleaved.getNumBands() >= CHANNELS) {
            var data = bytes.getData();
            var stride = interleaved.getScanlineStride();
            var pixelStride = interleaved.getPixelStride();
            var offsets = interleaved.getBandOffsets();
            rows(h, w, y -> {
                var source = origin + y * stride;
                var target = y * w * CHANNELS;
                for (var x = 0; x < w; x++) {
                    var at = source + x * pixelStride;
                    px[target + x * CHANNELS] = data[at + offsets[0]] & 0xFF;
                    px[target + x * CHANNELS + 1] = data[at + offsets[1]] & 0xFF;
                    px[target + x * CHANNELS + 2] = data[at + offsets[2]] & 0xFF;
                }
            });
            return new Raster(w, h, px);
        }

        if (sampleModel instanceof SinglePixelPackedSampleModel packed
                && buffer instanceof DataBufferInt ints && packed.getNumBands() >= CHANNELS) {
            var data = ints.getData();
            var stride = packed.getScanlineStride();
            var masks = packed.getBitMasks();
            var shifts = packed.getBitOffsets();
            if (masks[0] >>> shifts[0] != 0xFF || masks[1] >>> shifts[1] != 0xFF
                    || masks[2] >>> shifts[2] != 0xFF) return null;
            rows(h, w, y -> {
                var source = origin + y * stride;
                var target = y * w * CHANNELS;
                for (var x = 0; x < w; x++) {
                    var p = data[source + x];
                    px[target + x * CHANNELS] = (p >>> shifts[0]) & 0xFF;
                    px[target + x * CHANNELS + 1] = (p >>> shifts[1]) & 0xFF;
                    px[target + x * CHANNELS + 2] = (p >>> shifts[2]) & 0xFF;
                }
            });
            return new Raster(w, h, px);
        }
        return null;
    }

    public Raster resize(int w, int h) {
        return Resampler.resize(this, w, h);
    }

    public Raster crop(int x0, int y0, int x1, int y1) {
        var left = Math.max(0, Math.min(x0, width));
        var top = Math.max(0, Math.min(y0, height));
        var right = Math.max(left + 1, Math.min(x1, width));
        var bottom = Math.max(top + 1, Math.min(y1, height));
        var w = right - left;
        var h = bottom - top;
        var out = new float[w * h * CHANNELS];
        for (var y = 0; y < h; y++)
            System.arraycopy(px, ((top + y) * width + left) * CHANNELS, out, y * w * CHANNELS, w * CHANNELS);
        return new Raster(w, h, out);
    }

    /**
     * Rotates by a multiple of 90 degrees counter-clockwise. Source rows are independent — for a
     * quarter turn each one becomes a column of the output — so this splits across threads. The
     * orientation search does four of these on a whole page.
     */
    public Raster rotateCcw(int degrees) {
        var angle = ((degrees % 360) + 360) % 360;
        if (angle == 0) return this;
        var swapped = angle != 180;
        var w = swapped ? height : width;
        var h = swapped ? width : height;
        var out = new float[w * h * CHANNELS];
        rows(height, width, y -> {
            for (var x = 0; x < width; x++) {
                var tx = switch (angle) {
                    case 90 -> y;
                    case 180 -> width - 1 - x;
                    default -> height - 1 - y;
                };
                var ty = switch (angle) {
                    case 90 -> width - 1 - x;
                    case 180 -> height - 1 - y;
                    default -> x;
                };
                var from = (y * width + x) * CHANNELS;
                var to = (ty * w + tx) * CHANNELS;
                out[to] = px[from];
                out[to + 1] = px[from + 1];
                out[to + 2] = px[from + 2];
            }
        });
        return new Raster(w, h, out);
    }

    /** Runs {@code body} over every source row, splitting across threads only when it pays. */
    static void rows(int height, int width, java.util.function.IntConsumer body) {
        if ((long) height * width >= PARALLEL_FLOOR) Workers.split(height, body);
        else for (var y = 0; y < height; y++) body.accept(y);
    }
}
