package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Reads one line crop, with the recognizer for its script.
 *
 * <p>One session per script, created when a line of that script is first met, and shared by every
 * thread from then on. The maps are concurrent and creation is guarded, so {@link #recognize} may be
 * called from many threads at once — which is the point, since the recognizer input is
 * {@code [1, 3, 60, -1]} with the batch dimension a literal 1 and so cannot be batched. Lines are
 * independent, and concurrency over them is what this stage has instead of a batch.
 *
 * <p>{@link #preload} exists so a caller can pay the session cost up front rather than inside the
 * first page it reads.
 */
public final class TextRecognizer implements AutoCloseable {
    public static final int HEIGHT = 60, MIN_WIDTH = 16, STEP_DIVISOR = 4, SPACE_INDEX = 0;
    static final String DATA = "data", SEQ_LENGTHS = "seq_lengths", LOGSOFTMAX = "logsoftmax";

    final OrtEnvironment env;
    final ModelPaths paths;
    final Map<ScriptGroup, Model> models = new ConcurrentHashMap<>();
    final Map<ScriptGroup, Vocabulary> vocabularies = new ConcurrentHashMap<>();

    final boolean gpu;

    public TextRecognizer(OrtEnvironment env, ModelPaths paths, boolean gpu) {
        this.env = env;
        this.paths = paths;
        this.gpu = gpu;
    }

    /** Loads the session and vocabulary for each script now, so no line pays for it later. */
    public void preload(ScriptGroup... scripts) throws OrtException, IOException {
        for (var script : scripts) {
            if (!paths.has(script)) continue;
            model(script);
            vocabulary(script);
        }
    }

    Model model(ScriptGroup script) throws OrtException {
        var existing = models.get(script);
        if (existing != null) return existing;
        synchronized (models) {
            var again = models.get(script);
            if (again != null) return again;
            var created = new Model(env, paths.recognizer(script), gpu, Threads.recognizer());
            models.put(script, created);
            return created;
        }
    }

    Vocabulary vocabulary(ScriptGroup script) throws IOException {
        var existing = vocabularies.get(script);
        if (existing != null) return existing;
        synchronized (vocabularies) {
            var again = vocabularies.get(script);
            if (again != null) return again;
            var loaded = Vocabulary.load(paths.vocab(script), script.vocabSize);
            vocabularies.put(script, loaded);
            return loaded;
        }
    }

    public RecognizedLine recognize(Raster crop, ScriptGroup script) throws OrtException, IOException {
        var width = Math.max(MIN_WIDTH, (int) Math.round(HEIGHT * (double) crop.width / crop.height));
        var resized = crop.resize(width, HEIGHT);
        var session = model(script);
        var data = session.floats(Tensors.chw(resized, true, true), 1, Raster.CHANNELS, HEIGHT, width);
        var lengths = session.ints(new int[]{width / STEP_DIVISOR}, 1);
        try (data; lengths) {
            var out = session.run(Map.of(DATA, data, SEQ_LENGTHS, lengths), LOGSOFTMAX);
            return decode(out.get(LOGSOFTMAX), script, vocabulary(script));
        }
    }

    static RecognizedLine decode(Tensor logits, ScriptGroup script, Vocabulary vocabulary) {
        var vocabSize = logits.dim(logits.rank() - 1);
        var steps = logits.length() / Math.max(vocabSize, 1);
        var blank = script.blankIndex();
        var runs = new ArrayList<CharRun>();
        var previous = -1;

        for (var t = 0; t < steps; t++) {
            var base = t * vocabSize;
            var best = Tensors.argmax(logits.data, base, vocabSize);
            if (best != previous && best != blank) {
                var text = best == SPACE_INDEX ? " " : vocabulary.charAt(best);
                runs.add(new CharRun(text, Math.exp(logits.at(base + best)), t));
            }
            previous = best;
        }
        return new RecognizedLine(split(runs), steps);
    }

    static List<List<CharRun>> split(List<CharRun> runs) {
        var words = new ArrayList<List<CharRun>>();
        var current = new ArrayList<CharRun>();
        for (var run : runs) {
            if (run.text().isEmpty() || run.text().equals(" ")) {
                if (!current.isEmpty()) words.add(current);
                current = new ArrayList<>();
            } else {
                current.add(run);
            }
        }
        if (!current.isEmpty()) words.add(current);
        return words;
    }

    @Override
    public void close() {
        models.values().forEach(Model::close);
        models.clear();
    }
}
