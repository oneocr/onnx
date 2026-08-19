package oneocr.onnx;

public final class DetectionMaps {
    public static final int NEIGHBOURS = 8;

    public static final int CORNERS = 4, DELTA_CHANNELS = 8, ANCHOR_FACTOR = 8;

    final float[] scoresHorizontal, scoresVertical, linksHorizontal, linksVertical, deltasHorizontal, deltasVertical;
    final int rows, cols;
    final double scaleX, scaleY;
    double stride = TextDetector.STRIDE;

    public double anchor() {
        return stride * ANCHOR_FACTOR;
    }

    public double centreX(int col) {
        return col * stride + stride / 2;
    }

    public double centreY(int row) {
        return row * stride + stride / 2;
    }

    DetectionMaps(float[] scoresHorizontal, float[] scoresVertical, float[] linksHorizontal, float[] linksVertical,
                  float[] deltasHorizontal, float[] deltasVertical,
                  int rows, int cols, double scaleX, double scaleY) {
        this.scoresHorizontal = scoresHorizontal;
        this.scoresVertical = scoresVertical;
        this.linksHorizontal = linksHorizontal;
        this.linksVertical = linksVertical;
        this.deltasHorizontal = deltasHorizontal;
        this.deltasVertical = deltasVertical;
        this.rows = rows;
        this.cols = cols;
        this.scaleX = scaleX;
        this.scaleY = scaleY;
    }

    float[] deltas(boolean vertical) {
        return vertical ? deltasVertical : deltasHorizontal;
    }

    float delta(boolean vertical, int channel, int row, int col) {
        return deltas(vertical)[(channel * rows + row) * cols + col];
    }

    public boolean verticalLayout(float threshold) {
        return countAbove(scoresVertical, threshold) > countAbove(scoresHorizontal, threshold);
    }

    float[] scores(boolean vertical) {
        return vertical ? scoresVertical : scoresHorizontal;
    }

    float[] links(boolean vertical) {
        return vertical ? linksVertical : linksHorizontal;
    }

    float score(boolean vertical, int row, int col) {
        return scores(vertical)[row * cols + col];
    }

    float link(boolean vertical, int neighbour, int row, int col) {
        return links(vertical)[(neighbour * rows + row) * cols + col];
    }

    static int countAbove(float[] values, float threshold) {
        var n = 0;
        for (var v : values) if (v > threshold) n++;
        return n;
    }
}
