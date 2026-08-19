package oneocr.onnx;

public final class Geometry {
    private Geometry() {}

    public static double[] mapBack(double x, double y, int angle, int width, int height) {
        return switch (((angle % 360) + 360) % 360) {
            case 90 -> new double[]{width - y, x};
            case 180 -> new double[]{width - x, height - y};
            case 270 -> new double[]{y, height - x};
            default -> new double[]{x, y};
        };
    }

    public static Quad mapBack(double[] corners, int angle, int width, int height) {
        var out = new double[8];
        for (var i = 0; i < 4; i++) {
            var p = mapBack(corners[i * 2], corners[i * 2 + 1], angle, width, height);
            out[i * 2] = p[0];
            out[i * 2 + 1] = p[1];
        }
        return new Quad().x1(out[0]).y1(out[1]).x2(out[2]).y2(out[3]).x3(out[4]).y3(out[5]).x4(out[6]).y4(out[7]);
    }
}
