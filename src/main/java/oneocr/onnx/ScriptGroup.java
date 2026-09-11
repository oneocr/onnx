package oneocr.onnx;

import java.util.*;

public enum ScriptGroup {
    CJK(32632), CYRILLIC(548), LATIN(415), ARABIC(221), DEVANAGARI(237),
    GREEK(244), THAI(199), HEBREW(201), TAMIL(179);

    public final int vocabSize;

    ScriptGroup(int vocabSize) {
        this.vocabSize = vocabSize;
    }

    public String fileName() {
        return name().toLowerCase(Locale.ROOT);
    }

    public int blankIndex() {
        return vocabSize - 1;
    }

    public boolean joinsWithoutSpaces() {
        return this == CJK;
    }

    public static Optional<ScriptGroup> ofClassifierIndex(int index) {
        return index >= 1 && index <= values().length ? Optional.of(BY_CLASSIFIER_INDEX[index - 1]) : Optional.empty();
    }

    /** This script's index in the classifier's score vector; the inverse of {@link #ofClassifierIndex}. */
    public int classifierIndex() {
        for (var i = 0; i < BY_CLASSIFIER_INDEX.length; i++) if (BY_CLASSIFIER_INDEX[i] == this) return i + 1;
        return 0;
    }

    static final ScriptGroup[] BY_CLASSIFIER_INDEX =
            {CJK, CYRILLIC, LATIN, ARABIC, DEVANAGARI, GREEK, THAI, HEBREW, TAMIL};

    public static Optional<ScriptGroup> ofVocabSize(int size) {
        return Arrays.stream(values()).filter(s -> s.vocabSize == size).findFirst();
    }

    public static Optional<ScriptGroup> byName(String name) {
        var key = name.trim().toLowerCase(Locale.ROOT);
        return Arrays.stream(values()).filter(s -> s.fileName().equals(key)).findFirst();
    }
}
