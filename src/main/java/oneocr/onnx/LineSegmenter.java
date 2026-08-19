package oneocr.onnx;

import java.util.*;

public final class LineSegmenter {
    static final int[] DY = {-1, -1, -1, 0, 1, 1, 1, 0}, DX = {-1, 0, 1, 1, 1, 0, -1, -1};
    static final int MIN_PIXELS = 5;
    static final double MIN_SIDE = 6.0;

    private LineSegmenter() {}

    public static List<LineShape> segment(DetectionMaps maps, boolean vertical, float scoreThreshold, float linkThreshold) {
        var rows = maps.rows;
        var cols = maps.cols;
        var parent = new int[rows * cols];
        Arrays.fill(parent, -1);
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                if (maps.score(vertical, r, c) > scoreThreshold) parent[r * cols + c] = r * cols + c;

        for (var r = 0; r < rows; r++) {
            for (var c = 0; c < cols; c++) {
                if (parent[r * cols + c] < 0) continue;
                for (var n = 0; n < DetectionMaps.NEIGHBOURS; n++) {
                    var nr = r + DY[n];
                    var nc = c + DX[n];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                    if (parent[nr * cols + nc] < 0) continue;
                    if (maps.link(vertical, n, r, c) > linkThreshold) union(parent, r * cols + c, nr * cols + nc);
                }
            }
        }

        var groups = new HashMap<Integer, List<Integer>>();
        for (var i = 0; i < parent.length; i++)
            if (parent[i] >= 0) groups.computeIfAbsent(find(parent, i), k -> new ArrayList<>()).add(i);

        var out = new ArrayList<LineShape>();
        for (var group : groups.values()) {
            if (group.size() < MIN_PIXELS) continue;
            var shape = LineShapes.build(group, maps, vertical);
            if (shape != null) out.add(shape);
        }
        return out;
    }

    static int find(int[] parent, int i) {
        var root = i;
        while (parent[root] != root) root = parent[root];
        while (parent[i] != root) {
            var next = parent[i];
            parent[i] = root;
            i = next;
        }
        return root;
    }

    static void union(int[] parent, int a, int b) {
        var ra = find(parent, a);
        var rb = find(parent, b);
        if (ra != rb) parent[ra] = rb;
    }
}
