package oneocr.onnx;

import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * One ONNX session. Safe to share across threads: {@code OrtSession.run} is, and sharing one
 * session measured faster than a session per worker (453 ms against 684 ms over 99 lines), because
 * the weights and the arena are then loaded once.
 */
public final class Model implements AutoCloseable {
    final OrtEnvironment env;
    final OrtSession session;

    Model(OrtEnvironment env, Path file) throws OrtException {
        this(env, file, false, 0);
    }

    Model(OrtEnvironment env, Path file, boolean gpu) throws OrtException {
        this(env, file, gpu, 0);
    }

    Model(OrtEnvironment env, Path file, boolean gpu, int intraOpThreads) throws OrtException {
        this.env = env;
        try {
            this.session = env.createSession(file.toString(), Providers.options(gpu, intraOpThreads));
        } catch (OrtException e) {
            throw e;
        } catch (Exception e) {
            throw new IllegalStateException("could not open " + file, e);
        }
    }

    OnnxTensor floats(float[] data, long... shape) throws OrtException {
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
    }

    OnnxTensor ints(int[] data, long... shape) throws OrtException {
        return OnnxTensor.createTensor(env, IntBuffer.wrap(data), shape);
    }

    /**
     * Runs the session and copies out the named outputs. The names are also handed to ONNX Runtime
     * so it can prune whatever no wanted output depends on — worth about 5% on the detector, which
     * declares one output more than this pipeline reads.
     */
    Map<String, Tensor> run(Map<String, OnnxTensor> inputs, String... wanted) throws OrtException {
        try (var result = session.run(inputs, Set.of(wanted))) {
            var out = new LinkedHashMap<String, Tensor>();
            for (var name : wanted) out.put(name, copy(result, name));
            return out;
        }
    }

    static Tensor copy(OrtSession.Result result, String name) throws OrtException {
        var value = result.get(name).orElseThrow(() -> new IllegalStateException("model has no output " + name));
        var tensor = (OnnxTensor) value;
        var buffer = tensor.getFloatBuffer();
        var data = new float[buffer.remaining()];
        buffer.get(data);
        return new Tensor(data, tensor.getInfo().getShape());
    }

    @Override
    public void close() {
        try {
            session.close();
        } catch (Exception e) {
            throw new IllegalStateException("closing session failed", e);
        }
    }
}
