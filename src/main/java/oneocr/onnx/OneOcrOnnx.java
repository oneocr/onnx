package oneocr.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.*;

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

    public OcrResult recognize(Raster image, ScriptGroup fixedScript, Integer rotation) throws OrtException, IOException {
        require(image);
        var angle = rotation != null ? rotation : OrientationCorrector.estimate(image, detector, classifier, maxSide);
        var upright = image.rotateCcw(angle);
        var levels = detector.run(upright, maxSide);
        var vertical = levels.get(0).verticalLayout(scoreThreshold);

        var perLevel = new ArrayList<List<LineShape>>();
        for (var maps : levels) perLevel.add(LineSegmenter.segment(maps, vertical, scoreThreshold, linkThreshold));
        var shapes = LevelMerge.merge(perLevel);
        shapes.sort(vertical
                ? Comparator.comparingDouble((LineShape s) -> s.crop().x()).reversed()
                : Comparator.comparingDouble(s -> s.crop().y()));

        var lines = new ArrayList<OcrLine>();
        for (var shape : shapes.subList(0, Math.min(maxLines, shapes.size()))) {
            var line = readLine(image, upright, shape, vertical, fixedScript, angle);
            if (line != null) lines.add(line);
        }
        return new OcrResult().lines(lines).imageAngle(angle);
    }

    OcrLine readLine(Raster original, Raster upright, LineShape shape, boolean vertical, ScriptGroup fixedScript, int angle)
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
        var script = fixedScript != null ? fixedScript : detectScript(processed);
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

    ScriptGroup detectScript(Raster crop) throws OrtException {
        var verdict = classifier.classify(crop);
        return ScriptGroup.ofClassifierIndex(verdict.scriptIndex())
                .filter(paths::has)
                .orElse(ScriptGroup.LATIN);
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
