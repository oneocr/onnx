package oneocr.onnx;

import datapotter.datahelper.Data;

@Data
public final class OcrWord extends OcrWord_A {
    String text;
    Quad bounds;
    Double confidence;
}
