package oneocr.onnx;

/**
 * What the script classifier made of one line crop.
 *
 * @param scriptIndex the classifier's own index, 0 meaning "not a valid script" and 1..9 mapping
 *                    through {@link ScriptGroup#ofClassifierIndex}
 * @param flipScore   positive when the crop reads the right way up, negative when it is upside
 *                    down. The sign is what separates a correct orientation from a 180-degree one.
 * @param scores      the full score vector, kept so a caller restricting the classifier to a set of
 *                    candidate scripts can take the best of those rather than only the overall best
 */
record ScriptVerdict(int scriptIndex, float flipScore, float[] scores) {

    /** The score for one classifier index, or negative infinity if it is out of range. */
    float scoreAt(int index) {
        return index >= 0 && index < scores.length ? scores[index] : Float.NEGATIVE_INFINITY;
    }
}
