package oneocr.onnx;

record LineBox(double x, double y, double width, double height) {
    double right() {
        return x + width;
    }

    double bottom() {
        return y + height;
    }
}
