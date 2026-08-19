package oneocr.onnx;

import java.util.*;

public final class LevelMerge {
    static final double OVERLAP = 0.3;

    private LevelMerge() {}

    public static List<LineShape> merge(List<List<LineShape>> perLevel) {
        var all = new ArrayList<LineShape>();
        perLevel.forEach(all::addAll);
        all.sort(Comparator.comparingDouble(LineShape::score).reversed());

        var kept = new ArrayList<LineShape>();
        for (var candidate : all) {
            var clash = false;
            for (var accepted : kept) {
                if (containment(candidate.crop(), accepted.crop()) > OVERLAP) {
                    clash = true;
                    break;
                }
            }
            if (!clash) kept.add(candidate);
        }
        return kept;
    }

    static double containment(LineBox a, LineBox b) {
        var left = Math.max(a.x(), b.x());
        var top = Math.max(a.y(), b.y());
        var right = Math.min(a.right(), b.right());
        var bottom = Math.min(a.bottom(), b.bottom());
        if (right <= left || bottom <= top) return 0;
        var overlap = (right - left) * (bottom - top);
        var smaller = Math.min(a.width() * a.height(), b.width() * b.height());
        return smaller <= 0 ? 0 : overlap / smaller;
    }
}
