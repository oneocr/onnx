package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.*;

/**
 * The OneOCR pipeline over ONNX Runtime: detect lines, identify each line's script, recognise it.
 *
 * <p><b>{@link #recognize} is safe to call from several threads at once</b>, on one instance. The
 * sessions are shared — {@code OrtSession.run} is thread-safe, and sharing measured faster than a
 * session per thread — and the lazily created recognizer sessions are guarded. Configuration
 * ({@link #maxLines}, the thresholds) must be set before the first call and not changed while one
 * is running, and {@link #close} must not race a call.
 *
 * <p>Within one call, lines are read concurrently on a pool shared across the whole process; see
 * {@link Workers}. A caller that already parallelises over pages should set
 * {@code oneocr.threads.lines=1} so the two do not oversubscribe the machine, or leave it alone and
 * let the shared pool absorb both — it is one pool either way.
 *
 * <p>{@link #preload} loads every recognizer a caller might need up front, so the cost is paid at
 * startup rather than by whichever page first contains a given script.
 */
public final class OneOcrOnnx implements AutoCloseable {
    public static final int MIN_SIDE = 50, MAX_SIDE_LIMIT = 10000, CROP_PADDING = 2;

    final OrtEnvironment env = OrtEnvironment.getEnvironment();
    final ModelPaths paths;
    final TextDetector detector;
    final ScriptClassifier classifier;
    final TextRecognizer recognizer;

    int maxSide = TextDetector.MAX_SIDE, maxLines = 1000;
    float scoreThreshold = 0.5f, linkThreshold = 0.0f;
    double verticalGrowth = Double.parseDouble(System.getProperty("oneocr.growY", "0")),
            horizontalGrowth = Double.parseDouble(System.getProperty("oneocr.growX", "0"));

    public OneOcrOnnx(ModelPaths paths) throws OrtException {
        this(paths, false);
    }

    public OneOcrOnnx(ModelPaths paths, boolean gpu) throws OrtException {
        this.paths = paths;
        this.detector = new TextDetector(env, paths.detector(), gpu);
        this.classifier = new ScriptClassifier(env, paths.classifier(), gpu);
        this.recognizer = new TextRecognizer(env, paths, gpu);
    }

    public OneOcrOnnx maxLines(int value) {
        this.maxLines = value;
        return this;
    }

    /**
     * Loads the recognizer session and vocabulary for each named script now. With no arguments,
     * loads every script the model folder actually has. Scripts that are absent are skipped rather
     * than failing, since the shipped set depends on the Windows build the models came out of.
     */
    public OneOcrOnnx preload(ScriptGroup... scripts) throws OrtException, IOException {
        recognizer.preload(scripts.length == 0 ? ScriptGroup.values() : scripts);
        return this;
    }

    /**
     * Reads a page with everything detected and nothing assumed.
     *
     * @param fixedScript recognise every line with this script and never call the classifier, or
     *                    null to classify each line
     * @param rotation    the page's angle, or null to search for it
     */
    public OcrResult recognize(Raster image, ScriptGroup fixedScript, Integer rotation) throws OrtException, IOException {
        return recognize(image, OcrOptions.defaults().script(fixedScript).rotation(rotation));
    }

    /** Reads a page. See {@link OcrOptions} for what can be skipped, and what skipping it costs. */
    public OcrResult recognize(Raster image, OcrOptions options) throws OrtException, IOException {
        require(image);
        var angle = options.rotation() != null ? options.rotation()
                : OrientationCorrector.estimate(image, detector, classifier, maxSide, options.orientationSearch());
        var upright = image.rotateCcw(angle);
        var levels = detector.run(upright, maxSide);
        var vertical = levels.get(0).verticalLayout(scoreThreshold);

        var perLevel = new ArrayList<List<LineShape>>();
        for (var maps : levels) perLevel.add(LineSegmenter.segment(maps, vertical, scoreThreshold, linkThreshold));
        var shapes = LevelMerge.merge(perLevel);
        shapes.sort(vertical
                ? Comparator.comparingDouble((LineShape s) -> s.crop().x()).reversed()
                : Comparator.comparingDouble(s -> s.crop().y()));

        // Lines are independent, so they are read concurrently and the results reassembled in the
        // order the shapes were sorted into — a parallel run produces the same document as a
        // sequential one. This is the pipeline's largest single cost and the only stage with enough
        // work to fill the machine; the recognizer cannot be batched, as its input declares a batch
        // dimension of literally 1, so concurrency is what stands in for a batch here.
        var selected = shapes.subList(0, Math.min(maxLines, shapes.size()));
        var read = Workers.map(selected.size(),
                i -> readLine(image, upright, selected.get(i), vertical, options, angle));

        var lines = new ArrayList<OcrLine>(selected.size());
        for (var one : read) if (one != null) lines.add((OcrLine) one);
        return new OcrResult().lines(lines).imageAngle(angle);
    }

    OcrLine readLine(Raster original, Raster upright, LineShape shape, boolean vertical, OcrOptions options, int angle)
            throws OrtException, IOException {
        var box = shape.crop();
        var growY = (int) Math.round(box.height() * verticalGrowth);
        var growX = (int) Math.round(box.height() * horizontalGrowth);
        var left = Math.max(0, (int) box.x() - CROP_PADDING - growX);
        var top = Math.max(0, (int) box.y() - CROP_PADDING - growY);
        var right = Math.min(upright.width, (int) box.right() + CROP_PADDING + growX);
        var bottom = Math.min(upright.height, (int) box.bottom() + CROP_PADDING + growY);
        if (right - left < 1 || bottom - top < 1) return null;

        var crop = upright.crop(left, top, right, bottom);
        var processed = vertical ? crop.rotateCcw(90) : crop;
        var script = options.classifies() ? detectScript(processed, options) : options.script();
        var recognized = recognizer.recognize(processed, script);
        if (recognized.words().isEmpty()) return null;

        var words = words(recognized, processed.width, left, top, right, bottom, vertical, angle, original);
        var texts = words.stream().map(OcrWord::getText).toList();
        var text = String.join(script.joinsWithoutSpaces() ? "" : " ", texts);
        if (text.isBlank()) return null;

        var q = shape.quad();
        var bounds = Geometry.mapBack(new double[]{q.getX1(), q.getY1(), q.getX2(), q.getY2(),
                q.getX3(), q.getY3(), q.getX4(), q.getY4()}, angle, original.width, original.height);
        return new OcrLine().text(text).bounds(bounds).vertical(vertical).script(script.fileName()).words(words);
    }

    ScriptGroup detectScript(Raster crop, OcrOptions options) throws OrtException {
        var verdict = classifier.classify(crop);
        if (!options.candidates().isEmpty()) return bestCandidate(verdict, options.candidates());
        return ScriptGroup.ofClassifierIndex(verdict.scriptIndex())
                .filter(paths::has)
                .orElse(ScriptGroup.LATIN);
    }

    /**
     * The highest scoring of the allowed scripts, rather than the highest scoring of all ten.
     *
     * <p>This is what a caller who knows the document is, say, Latin and CJK should use. The
     * classifier's raw label is noisy — on a rendered English page it sent 21 of 23 lines to the
     * Cyrillic recognizer — and confining it to what can actually occur fixes the misrouting without
     * forcing one script on every line, which is what breaks a mixed document.
     */
    ScriptGroup bestCandidate(ScriptVerdict verdict, Set<ScriptGroup> candidates) {
        ScriptGroup best = null;
        var bestScore = Float.NEGATIVE_INFINITY;
        for (var candidate : candidates) {
            if (!paths.has(candidate)) continue;
            var score = verdict.scoreAt(candidate.classifierIndex());
            if (best == null || score > bestScore) {
                bestScore = score;
                best = candidate;
            }
        }
        return best != null ? best : ScriptGroup.LATIN;
    }

    List<OcrWord> words(RecognizedLine recognized, int processedWidth, int left, int top, int right, int bottom,
                        boolean vertical, int angle, Raster original) {
        var out = new ArrayList<OcrWord>();
        var perStep = (double) processedWidth / Math.max(recognized.steps(), 1);
        for (var word : recognized.words()) {
            var start = word.get(0).step() * perStep;
            var end = (word.get(word.size() - 1).step() + 1) * perStep;
            var corners = vertical
                    ? new double[]{left, top + start, right, top + start, right, top + end, left, top + end}
                    : new double[]{left + start, top, left + end, top, left + end, bottom, left + start, bottom};
            var text = word.stream().map(CharRun::text).reduce("", String::concat);
            var confidence = word.stream().mapToDouble(CharRun::probability).average().orElse(0);
            out.add(new OcrWord().text(text).confidence(confidence)
                    .bounds(Geometry.mapBack(corners, angle, original.width, original.height)));
        }
        return out;
    }

    static void require(Raster image) {
        if (image.width < MIN_SIDE || image.height < MIN_SIDE || image.width > MAX_SIDE_LIMIT || image.height > MAX_SIDE_LIMIT)
            throw new IllegalArgumentException("image " + image.width + "x" + image.height
                    + " is outside the supported range " + MIN_SIDE + " to " + MAX_SIDE_LIMIT + " px");
    }

    @Override
    public void close() {
        recognizer.close();
        classifier.close();
        detector.close();
    }
}
