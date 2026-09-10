package oneocr.onnx;

import datapotter.datahelper.Data;

import java.util.List;

@Data
public final class OcrLine extends OcrLine_A {
    String text;
    Quad bounds;
    Boolean vertical;
    String script;
    List<OcrWord> words = List.of();
}
