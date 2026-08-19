# oneocr-onnx

The Windows 11 OneOCR pipeline, reimplemented in pure Java over ONNX Runtime.

Runs on **Linux, macOS and Windows**. No `oneocr.dll`, no FFM, no Python, no OpenCV.

Windows 11's Snipping Tool ships a compact, very good OCR engine. It lives in `oneocr.dll` plus an encrypted model container, and it has always been Windows-only. It isn't any more: the container holds eleven ordinary ONNX networks, and everything around them is arithmetic that Java can do.

Verified against the real DLL on the same images: output is byte-identical between Windows and Linux, and on rendered samples the text matches Microsoft's own output character for character for Latin, Cyrillic, Greek, Devanagari and Tamil.

## What it does

- Text detection over a three-level feature pyramid, with the quadrilateral regression the models actually provide, so lines come back as rotated quads rather than axis-aligned boxes.
- Script classification across nine scripts: CJK, Cyrillic, Latin, Arabic, Devanagari, Greek, Thai, Hebrew, Tamil.
- CTC recognition with a per-script recognizer and vocabulary.
- Page orientation estimation, vertical (CJK) layout handling, per-word confidence.

## You need the models first

The models are not in this repository and never will be. They are Microsoft's, and they are encrypted inside `oneocr.onemodel` on a licensed Windows installation.

Run [`oneocr/modelex`](https://github.com/oneocr/modelex) **once on Windows** to extract them from your own machine. That writes `~/oneocr/models/`. Copy that folder to wherever you want to run OCR — after that, Windows is out of the picture.

## Use

```
java -jar oneocr-onnx.jar page.png
java -jar oneocr-onnx.jar page.png --lang devanagari --rotation 0 --detail
```

As a library:

```java
try (var engine = new OneOcrOnnx(ModelPaths.resolve(Path.of("/opt/oneocr/models")))) {
    var result = engine.recognize(Raster.of(image), null, 0);
    System.out.println(result.fullText());
}
```

`recognize` takes an optional `ScriptGroup` to force a script and an optional rotation. Pass `0` for pages you already know are upright — the orientation search costs about 750 ms per page because it runs the detector on all four rotations.

Requires JDK 22 or later.

## Speed

On a dense 105-line book page, warm, rotation known: **about 2.6 s per page on CPU**, against roughly 3.1 s for Microsoft's own DLL on the same page. So it is not a compromise you accept in order to leave Windows.

Recognition is about 85% of that, one session call per line. Batching those calls and parallelising across lines is the obvious next work and is tracked in the project's PRP 21.

A CUDA attempt made it **2.4× slower**, because the recognizer is called once per line with a different input width each time and ONNX Runtime re-selects convolution algorithms per shape. Fixed-width batching has to come first; the GPU is not a switch you can flip. The measurements and diagnosis are recorded in PRP 21.

## Known gaps

- **Hebrew and Arabic** come out in visual rather than logical order. Right-to-left needs bidi handling that isn't written yet.
- **Tamil** auto-detects as Thai. The recognizer is fine — pass `--lang tamil` and the output matches the DLL exactly. It's the classifier that needs work.
- **Multi-column pages** interleave lines; there is no column awareness in this module.
- The mapping from vocabulary size to script is specific to the OneOCR build it was extracted from. A different Windows build could ship a different set.

## Deployment notes

- `OneOcrOnnx` is **not thread-safe**: recognizer sessions and vocabularies are populated lazily without synchronisation. One instance per worker thread, or a lock.
- ONNX Runtime unpacks its native library into `java.io.tmpdir`. A read-only `/tmp` fails at class-load time with a confusing error.
- It links against glibc, so plain Alpine/musl needs extra work.
- Recognizers load lazily per script, so you only need to ship the scripts you use — Latin alone is 6.3 MB of the 39 MB.
- The ONNX Runtime jar carries natives for every platform. Keeping only `ai/onnxruntime/native/linux-x64/` trims about 118 MB from a container image.

## Credits

This exists because other people did the hard part first, in the open.

- **[b1tg/win11-oneocr](https://github.com/b1tg/win11-oneocr)** — the first person to reverse engineer the engine and publish it. Everything downstream, including this, starts there.
- **[bropines/oneocr-onnx-python](https://github.com/bropines/oneocr-onnx-python)** — decomposed `oneocr.onemodel` into its eleven ONNX sub-models by hooking the ONNX Runtime C API in memory, reimplemented the pipeline in Python, and wrote it all up honestly. This module is a Java descendant of that work and would not exist without it. Their writeup is the best explanation of the extraction that exists.
- **[JanikRitz/win11-oneocr](https://github.com/JanikRitz/win11-oneocr)** — a fuller C++ implementation.
- **[AuroraWright/oneocr](https://github.com/AuroraWright/oneocr)** and **[Cecilia-pj/win11_oneocr_py](https://github.com/Cecilia-pj/win11_oneocr_py)** — Python implementations that clarified the FFI surface.
- **[wangfu91/oneocr-rs](https://github.com/wangfu91/oneocr-rs)** — Rust bindings, and a cleaner shallow API design than most.
- **MattyMroz** — an early decryption effort, referenced in the issue threads.
- **Microsoft** — the engine and the models, which remain theirs. Nothing of theirs is redistributed here.

Where this repository differs from its ancestors, and why, is documented in the project's PRP 04: the detector's unused `bbox_deltas` outputs turn out to be an 8-channel quadrilateral regression scaled by 8× the level stride; three of the reference project's script names were mismatched (V=179 is Tamil, not Greek; V=244 is Greek, not Hebrew; V=201 is Hebrew, not Bengali, and this build ships no Bengali recognizer at all); confidences here are `exp(logsoftmax)` rather than a softmax applied on top of log-probabilities; and the rotation inverse is derived from the forward transform rather than being a mirrored copy.

## Licence

The Java code here is ours. The models are Microsoft's and are neither included nor redistributed. Extract them from your own licensed Windows installation.
