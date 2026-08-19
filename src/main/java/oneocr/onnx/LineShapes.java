package oneocr.onnx;

import java.util.*;

public final class LineShapes {
    private LineShapes() {}

    static LineShape build(List<Integer> cells, DetectionMaps maps, boolean vertical) {
        var anchor = maps.anchor();
        var cols = maps.cols;
        var corners = new HashMap<Integer, double[]>(cells.size());
        var score = 0.0;

        for (var index : cells) {
            var r = index / cols;
            var c = index % cols;
            var cx = maps.centreX(c);
            var cy = maps.centreY(r);
            var quad = new double[DetectionMaps.DELTA_CHANNELS];
            for (var k = 0; k < DetectionMaps.CORNERS; k++) {
                quad[k * 2] = (cx + maps.delta(vertical, k * 2, r, c) * anchor) / maps.scaleX;
                quad[k * 2 + 1] = (cy + maps.delta(vertical, k * 2 + 1, r, c) * anchor) / maps.scaleY;
            }
            corners.put(index, quad);
            score += maps.score(vertical, r, c);
        }

        var bounds = bounds(corners.values());
        if (bounds.width() < LineSegmenter.MIN_SIDE || bounds.height() < LineSegmenter.MIN_SIDE) return null;

        var ordered = new ArrayList<>(cells);
        ordered.sort(Comparator.comparingDouble(i -> vertical
                ? maps.centreY(i / cols) : maps.centreX(i % cols)));
        var first = corners.get(ordered.get(0));
        var last = corners.get(ordered.get(ordered.size() - 1));
        var quad = vertical ? spanVertical(first, last) : spanHorizontal(first, last);
        return new LineShape(quad, bounds, score / cells.size(), vertical);
    }

    static Quad spanHorizontal(double[] first, double[] last) {
        return new Quad().x1(first[0]).y1(first[1]).x2(last[2]).y2(last[3])
                .x3(last[4]).y3(last[5]).x4(first[6]).y4(first[7]);
    }

    static Quad spanVertical(double[] first, double[] last) {
        return new Quad().x1(first[0]).y1(first[1]).x2(first[2]).y2(first[3])
                .x3(last[4]).y3(last[5]).x4(last[6]).y4(last[7]);
    }

    static LineBox bounds(Collection<double[]> quads) {
        var minX = Double.MAX_VALUE;
        var minY = Double.MAX_VALUE;
        var maxX = -Double.MAX_VALUE;
        var maxY = -Double.MAX_VALUE;
        for (var q : quads) {
            for (var k = 0; k < DetectionMaps.CORNERS; k++) {
                minX = Math.min(minX, q[k * 2]);
                maxX = Math.max(maxX, q[k * 2]);
                minY = Math.min(minY, q[k * 2 + 1]);
                maxY = Math.max(maxY, q[k * 2 + 1]);
            }
        }
        return new LineBox(minX, minY, maxX - minX, maxY - minY);
    }
}
