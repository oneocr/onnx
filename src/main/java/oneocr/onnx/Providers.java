package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public final class Providers {
    private Providers() {}

    public static OrtSession.SessionOptions options(boolean gpu) throws Exception {
        var options = new OrtSession.SessionOptions();
        if (gpu) attach(options);
        return options;
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
