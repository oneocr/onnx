package oneocr.onnx;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * What to do, and what to skip, for one call to {@link OneOcrOnnx#recognize}.
 *
 * <p>{@link #defaults()} is exactly what the engine has always done: search all four orientations,
 * and let the classifier pick a script for every line. Every option here is an opt-in that trades
 * some of that work away for speed, and each one says plainly what it costs. Nothing in this class
 * changes behaviour until a caller asks for it.
 *
 * <p>The two that matter, measured on this project's fixtures:
 * <ul>
 *   <li>{@link #rotation(int)} — 1.4x to 4.3x, and free of any accuracy cost when the angle really
 *       is known. Largest on big, text-sparse pages, where four detector calls over a whole page
 *       dwarf the handful of lines to read. A page rendered from a PDF is upright: pass 0.
 *   <li>{@link #script(ScriptGroup)} — 1.5x to 1.7x, and free of accuracy cost on a document whose
 *       language is known. On a rendered English page the classifier routed 21 of 23 lines to the
 *       CYRILLIC recognizer and the text came out the same either way, because the recognizers share
 *       most of their glyphs. It is NOT free on a mixed document: forcing latin on a bilingual
 *       Chinese page destroyed every CJK line.
 * </ul>
 *
 * <p>{@link #candidates(ScriptGroup...)} is the middle road for a document that is mixed but not
 * arbitrary — it keeps per-line classification and confines it to the scripts that can actually
 * occur, which fixes misrouting without forcing one answer on the whole page.
 *
 * <p>Immutable; every method returns a new instance, so one configured constant can be shared
 * across threads and pages.
 */
public record OcrOptions(Integer rotation, OrientationSearch orientationSearch,
                         ScriptGroup script, Set<ScriptGroup> candidates) {

    static final OcrOptions DEFAULTS =
            new OcrOptions(null, OrientationSearch.FULL, null, Set.of());

    public OcrOptions {
        candidates = candidates == null || candidates.isEmpty()
                ? Set.of() : Collections.unmodifiableSet(EnumSet.copyOf(candidates));
        if (orientationSearch == null) orientationSearch = OrientationSearch.FULL;
    }

    /** Everything detected, nothing assumed. What the engine did before any of this existed. */
    public static OcrOptions defaults() {
        return DEFAULTS;
    }

    /** The page is at this angle; skip the search entirely. Must be 0, 90, 180 or 270. */
    public OcrOptions rotation(Integer degrees) {
        if (degrees != null && degrees != 0 && degrees != 90 && degrees != 180 && degrees != 270)
            throw new IllegalArgumentException("rotation must be 0, 90, 180 or 270, not " + degrees);
        return new OcrOptions(degrees, orientationSearch, script, candidates);
    }

    /** The page is upright. The right call for anything rendered from a PDF. */
    public OcrOptions upright() {
        return rotation(0);
    }

    /** How hard to look for the orientation, when it is not given. */
    public OcrOptions orientationSearch(OrientationSearch search) {
        return new OcrOptions(rotation, search, script, candidates);
    }

    /**
     * Read every line with this script's recognizer and never call the classifier. Correct when the
     * document's language is known; damaging when it is not — see the class notes.
     */
    public OcrOptions script(ScriptGroup only) {
        return new OcrOptions(rotation, orientationSearch, only, candidates);
    }

    /**
     * Confine the classifier to these scripts, keeping it per line. Pass nothing to allow them all.
     * A script with no model in the folder is dropped, so an empty result falls back to detection.
     */
    public OcrOptions candidates(ScriptGroup... allowed) {
        return new OcrOptions(rotation, orientationSearch, script,
                allowed.length == 0 ? Set.of() : EnumSet.copyOf(Set.of(allowed)));
    }

    /** Whether the classifier has to run at all for a line. */
    public boolean classifies() {
        return script == null;
    }
}
