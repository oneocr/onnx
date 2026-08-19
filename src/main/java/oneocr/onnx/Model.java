package oneocr.onnx;

import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class Model implements AutoCloseable {
    final OrtEnvironment env;
    final OrtSession session;

    Model(OrtEnvironment env, Path file) throws OrtException {
        this(env, file, false);
    }

    Model(OrtEnvironment env, Path file, boolean gpu) throws OrtException {
        this.env = env;
        try {
            this.session = env.createSession(file.toString(), Providers.options(gpu));
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

    Map<String, Tensor> run(Map<String, OnnxTensor> inputs, String... wanted) throws OrtException {
        try (var result = session.run(inputs)) {
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
