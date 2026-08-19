package oneocr.onnx;

import java.util.List;

record RecognizedLine(List<List<CharRun>> words, int steps) {}
