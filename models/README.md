# Parcae models

- `models/python/parcae_model.pth`: PyTorch checkpoint produced by the existing Python training loop.
- `models/rust/parcae_model.onnx`: ONNX export used by Rust inference (native and WASM builds via onnxruntime).
- `models/rust/book/centurion_book_v1.bin`: Centurion opening book artifact (self-play generated).
- `models/rust/tablebases/centurion_tb_4pc_v1.bin`: Centurion tablebase artifact for up to 4 pieces.
- `models/rust/nnue/centurion_nnue_v1.bin`: Centurion NNUE weights (optional but recommended).

Export flow:
1. Train in Python to update `parcae_model.pth`.
2. Run `scripts/export_onnx.py` to emit the ONNX file into `models/rust/`.
3. Configure the Rust server or WASM AI to read `PARCAE_MODEL_PATH` (defaults to `models/rust/parcae_model.onnx`).

Centurion artifact generation (from `rust/`):

```bash
cargo run --release -p ai --bin centurion_bookgen -- --games 5000 --max-plies 220 --explore-plies 14 --epsilon 0.35 --threads 8 --out ../models/rust/book/centurion_book_v1.bin
cargo run --release -p ai --bin centurion_tbgen -- --games 20000 --max-plies 320 --max-pieces 4 --synthetic 40000 --explore-plies 28 --epsilon 0.45 --threads 8 --out ../models/rust/tablebases/centurion_tb_4pc_v1.bin
cargo run --release -p ai --bin centurion_nnue_train -- --samples 60000 --epochs 10 --hidden 128 --threads 8 --out ../models/rust/nnue/centurion_nnue_v1.bin
```
