package oneocr.onnx;

import datapotter.datahelper.Data;

import java.util.List;
import java.util.stream.Collectors;

@Data
public final class OcrResult extends OcrResult_A {
    List<OcrLine> lines = List.of();
    Integer imageAngle = 0;

    public String fullText() {
        return lines.stream().map(OcrLine::getText).collect(Collectors.joining("\n"));
    }
}
