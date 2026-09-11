package oneocr.onnx;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;

/**
 * Asserts that {@link Raster#direct} and {@link Raster#viaColorModel} produce the same floats.
 *
 * <p>The direct read is only sound where the image's storage is plain sRGB bands; for anything else
 * — grayscale, a non-sRGB colour space, premultiplied alpha, an indexed palette — the colour model
 * has to do the conversion and the direct path must decline. This checks both halves of that claim:
 * where a fast path is taken it agrees exactly, and where it must not be taken it is not offered.
 */
public class RasterProbe {
    record Case(String name, int type, boolean expectDirect) {}

    static final List<Case> CASES = List.of(
            new Case("TYPE_3BYTE_BGR", BufferedImage.TYPE_3BYTE_BGR, true),
            new Case("TYPE_INT_RGB", BufferedImage.TYPE_INT_RGB, true),
            new Case("TYPE_INT_BGR", BufferedImage.TYPE_INT_BGR, true),
            new Case("TYPE_INT_ARGB", BufferedImage.TYPE_INT_ARGB, true),
            new Case("TYPE_4BYTE_ABGR", BufferedImage.TYPE_4BYTE_ABGR, true),
            new Case("TYPE_BYTE_GRAY", BufferedImage.TYPE_BYTE_GRAY, false),
            new Case("TYPE_USHORT_GRAY", BufferedImage.TYPE_USHORT_GRAY, false),
            new Case("TYPE_BYTE_INDEXED", BufferedImage.TYPE_BYTE_INDEXED, false),
            new Case("TYPE_INT_ARGB_PRE", BufferedImage.TYPE_INT_ARGB_PRE, false));

    public static void main(String[] args) throws Exception {
        var failures = 0;
        for (var one : CASES) failures += check(one, synthetic(one.type())) ? 0 : 1;

        for (var file : args) {
            var image = ImageIO.read(Path.of(file).toFile());
            var name = Path.of(file).getFileName() + " (BufferedImage type " + image.getType() + ")";
            failures += check(new Case(name, image.getType(), Raster.direct(image) != null), image) ? 0 : 1;
        }
        System.out.println(failures == 0 ? "PASS" : "FAIL: " + failures + " case(s)");
        System.exit(failures == 0 ? 0 : 1);
    }

    static boolean check(Case one, BufferedImage image) {
        var direct = Raster.direct(image);
        if (direct == null) {
            var ok = !one.expectDirect();
            System.out.printf("  %-42s no fast path%s%n", one.name(), ok ? "" : "  <-- EXPECTED ONE");
            return ok;
        }
        if (!one.expectDirect()) {
            System.out.printf("  %-42s fast path taken  <-- SHOULD HAVE DECLINED%n", one.name());
            return false;
        }
        var reference = Raster.viaColorModel(image);
        var worst = 0f;
        var at = -1;
        for (var i = 0; i < reference.px.length; i++) {
            var delta = Math.abs(reference.px[i] - direct.px[i]);
            if (delta > worst) {
                worst = delta;
                at = i;
            }
        }
        System.out.printf("  %-42s fast path, %,d floats, max delta %.0f%s%n",
                one.name(), reference.px.length, worst, worst == 0 ? "" : " at index " + at + "  <-- MISMATCH");
        return worst == 0;
    }

    /** A gradient with every channel varying independently, so a swapped band cannot hide. */
    static BufferedImage synthetic(int type) {
        var image = new BufferedImage(67, 43, type);
        for (var y = 0; y < image.getHeight(); y++)
            for (var x = 0; x < image.getWidth(); x++)
                image.setRGB(x, y, 0xFF000000 | (x * 3 % 256) << 16 | (y * 5 % 256) << 8 | (x + y) * 2 % 256);
        return image;
    }
}
