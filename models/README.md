# Parcae models

- `models/python/parcae_model.pth`: PyTorch checkpoint produced by the existing Python training loop.
- `models/rust/parcae_model.onnx`: ONNX export used by Rust inference (native and WASM builds via onnxruntime).

Export flow:
1. Train in Python to update `parcae_model.pth`.
2. Run `scripts/export_onnx.py` to emit the ONNX file into `models/rust/`.
3. Configure the Rust server or WASM AI to read `PARCAE_MODEL_PATH` (defaults to `models/rust/parcae_model.onnx`).
