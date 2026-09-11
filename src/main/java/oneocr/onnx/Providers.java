package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;

public final class Providers {
    /**
     * How loud ONNX Runtime is about these particular models. Left alone it writes several graph
     * warnings per session to stderr as UTF-16 — harmless, and under a service manager it lands in
     * the journal as mojibake, a screenful per model loaded. The warnings are about the models
     * themselves and say nothing a caller can act on: an initializer that also appears as a graph
     * input, and a conditional detector output that legitimately comes back with a placeholder
     * shape.
     *
     * <p>This is set per session rather than on the environment, which is a process-wide singleton
     * whose first caller wins. Setting it there would mean either losing to a caller that
     * configured it, or making that caller's own call warn about a changed name. Per session,
     * nobody is fought with. Override with {@code -Doneocr.ort.loglevel=WARNING} (or FATAL, INFO,
     * VERBOSE) to hear it all again.
     */
    public static final String LOG_LEVEL = "oneocr.ort.loglevel";

    private Providers() {}

    public static OrtSession.SessionOptions options(boolean gpu) throws Exception {
        return options(gpu, 0);
    }

    /**
     * @param intraOpThreads threads ONNX Runtime may use inside one operator, or 0 to leave its
     *                       default in place. See {@link Threads} for why the default is wrong for
     *                       every stage of this pipeline.
     */
    public static OrtSession.SessionOptions options(boolean gpu, int intraOpThreads) throws Exception {
        var options = new OrtSession.SessionOptions();
        options.setSessionLogLevel(logLevel());
        if (intraOpThreads > 0) options.setIntraOpNumThreads(intraOpThreads);
        if (gpu) attach(options);
        return options;
    }

    static OrtLoggingLevel logLevel() {
        var name = System.getProperty(LOG_LEVEL);
        if (name == null || name.isBlank()) return OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        try {
            return OrtLoggingLevel.valueOf("ORT_LOGGING_LEVEL_" + name.trim().toUpperCase(java.util.Locale.ROOT));
        } catch (IllegalArgumentException unknown) {
            return OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
        }
    }

    static void attach(OrtSession.SessionOptions options) {
        var available = OrtEnvironment.getAvailableProviders().toString();
        try {
            options.addCUDA(0);
            System.err.println("execution provider: CUDA");
            return;
        } catch (Throwable cuda) {
            System.err.println("CUDA unavailable (" + short_(cuda) + ")");
        }
        try {
            options.addDirectML(0);
            System.err.println("execution provider: DirectML");
            return;
        } catch (Throwable directml) {
            System.err.println("DirectML unavailable (" + short_(directml) + ")");
        }
        System.err.println("falling back to CPU. providers in this build: " + available
                + ". For CUDA, build with -Pgpu and install the CUDA 12 runtime plus cuDNN 9.");
    }

    static String short_(Throwable t) {
        var message = t.getMessage();
        return message == null ? t.getClass().getSimpleName() : message.split("\\R")[0];
    }
}
