package oneocr.onnx;

/** How hard {@link OneOcrOnnx} works to find out which way up a page is. */
public enum OrientationSearch {
    /**
     * Detect at all four quarter turns and take the best. Four detector calls, and the default,
     * because it is what has always been done and what every existing output was produced with.
     */
    FULL,

    /**
     * Try upright first and stop there if it looks upright, which costs one detector call instead of
     * four. A page that is actually rotated fails the test and falls through to {@link #FULL}, so
     * nothing is lost on those beyond the one call already made.
     *
     * <p>The test is that all sampled crops classify as a valid script AND the classifier's mean
     * flip score is positive. Measured over six images, the correct angle scored a positive flip
     * score every time (+3 to +5) while the 180-degrees-off angle scored a negative one (-4 to -6)
     * despite also having every sample land on a valid script — so it is the SIGN that separates
     * them, not the script count.
     *
     * <p>Opt in deliberately. This changes which angle can be chosen, and the evidence for it is six
     * images, three of them rotations of one page. It is a good bet on a corpus of upright pages and
     * it has not been validated on a large one. If the rotation is genuinely known, pass it instead:
     * that is both faster and certain.
     */
    EARLY_EXIT
}
