package oneocr.onnx;

import java.awt.image.BufferedImage;

public final class Raster {
    public static final int CHANNELS = 3;

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

    public static Raster of(BufferedImage image) {
        var w = image.getWidth();
        var h = image.getHeight();
        var argb = image.getRGB(0, 0, w, h, null, 0, w);
        var px = new float[w * h * CHANNELS];
        for (var i = 0; i < argb.length; i++) {
            var p = argb[i];
            px[i * CHANNELS] = (p >> 16) & 0xFF;
            px[i * CHANNELS + 1] = (p >> 8) & 0xFF;
            px[i * CHANNELS + 2] = p & 0xFF;
        }
        return new Raster(w, h, px);
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

    public Raster rotateCcw(int degrees) {
        var angle = ((degrees % 360) + 360) % 360;
        if (angle == 0) return this;
        var swapped = angle != 180;
        var w = swapped ? height : width;
        var h = swapped ? width : height;
        var out = new float[w * h * CHANNELS];
        for (var y = 0; y < height; y++) {
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
        }
        return new Raster(w, h, out);
    }
}
