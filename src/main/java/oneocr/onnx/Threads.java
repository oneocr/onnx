package oneocr.onnx;

/**
 * How many threads each part of the pipeline gets.
 *
 * <p>ONNX Runtime's own default is one intra-op thread per logical processor, and measurement says
 * that is wrong for every stage here, in both directions. The detector is one large call per page
 * and wants several threads, but not one per hyperthread — on an 8-core/16-thread machine it ran
 * 603 ms at the default, 334 ms at 8 threads and 591 ms again at 16. The recognizer and the script
 * classifier are called once per line on inputs far too small to divide, so they want one thread
 * each and the parallelism belongs at the line level instead, where 99 lines went from 1962 ms to
 * about 500 ms.
 *
 * <p>Every value can be overridden with a system property, because the right numbers depend on the
 * machine and on whether the caller is already running pages in parallel. A caller with its own
 * pool should set {@code oneocr.threads.lines=1} and keep the cores for itself.
 */
public final class Threads {
    public static final String LINES = "oneocr.threads.lines",
            DETECTOR = "oneocr.threads.detector",
            RECOGNIZER = "oneocr.threads.recognizer",
            CLASSIFIER = "oneocr.threads.classifier";

    private Threads() {}

    /** Workers reading lines concurrently. One disables the line pool and runs them in order. */
    public static int lines() {
        return property(LINES, processors());
    }

    /**
     * Intra-op threads for the detector. Half the logical processors approximates the physical core
     * count, which is where this stage peaked; a machine without SMT loses a little and never much.
     */
    public static int detector() {
        return property(DETECTOR, Math.max(1, processors() / 2));
    }

    public static int recognizer() {
        return property(RECOGNIZER, 1);
    }

    public static int classifier() {
        return property(CLASSIFIER, 1);
    }

    static int processors() {
        return Runtime.getRuntime().availableProcessors();
    }

    static int property(String name, int fallback) {
        var raw = System.getProperty(name);
        if (raw == null || raw.isBlank()) return fallback;
        try {
            return Math.max(1, Integer.parseInt(raw.trim()));
        } catch (NumberFormatException malformed) {
            return fallback;
        }
    }
}
