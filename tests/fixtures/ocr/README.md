# OCR Fixtures

`trocr-hello.png` contains the printed text `HELLO` as black text on a white background. It is used by the opt-in TrOCR ONNX acceptance test to verify semantic OCR output rather than only non-empty model output.

The fixture was generated once with Pillow using DejaVu Sans Bold at 240×80 and copied from the reviewed pre-split `rust-packages` fixture. Keep OCR fixtures small and intentional; do not replace them with generated model output.
