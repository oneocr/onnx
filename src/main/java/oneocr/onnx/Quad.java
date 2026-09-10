package oneocr.onnx;

import datapotter.datahelper.Data;

@Data
public final class Quad extends Quad_A {
    double x1, y1, x2, y2, x3, y3, x4, y4;

    public static Quad rect(double left, double top, double right, double bottom) {
        return new Quad().x1(left).y1(top).x2(right).y2(top).x3(right).y3(bottom).x4(left).y4(bottom);
    }

    public double left() {
        return Math.min(Math.min(x1, x2), Math.min(x3, x4));
    }

    public double top() {
        return Math.min(Math.min(y1, y2), Math.min(y3, y4));
    }

    public double width() {
        return Math.max(Math.max(x1, x2), Math.max(x3, x4)) - left();
    }

    public double height() {
        return Math.max(Math.max(y1, y2), Math.max(y3, y4)) - top();
    }
}
