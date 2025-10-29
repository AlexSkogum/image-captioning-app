# Image Captioning — Starter Project

This repository is a starter implementation for an image-captioning pipeline (CNN encoder + LSTM decoder with attention). It contains training, inference, a FastAPI service, and a minimal Gradio web UI.

Sections:
- Quick setup
- Training (small test)
- Run API + UI
- Deployment notes

See `configs/config.yaml` for default hyperparameters.

## Quick setup

1. Create and activate virtualenv (macOS / zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Kaggle (download MS COCO) — place keys in `~/.kaggle/kaggle.json` and follow Kaggle docs. Example (shell):

```bash
# Install kaggle CLI if needed
pip install kaggle
# Example to download COCO (requires proper kaggle dataset path and agreement)
# kaggle competitions download -c some-coco-dataset
```

3. Configs: edit `configs/config.yaml` for hyperparameters and dataset paths.

## Run a quick CPU test (small dataset)

1. Prepare a tiny dataset (CSV with columns `image_path,caption`) or use `web/sample_images/`.
2. Build vocab and preprocess:

```bash
python -m src.data --build-vocab --captions-file small_captions.csv --out-dir data/
```

3. Train for 1 epoch (CPU):

```bash
python -m src.train --config configs/config.yaml --dev
```

## Run API locally

1. Start API (after creating or using a checkpoint):

```bash
uvicorn src.api.main:app --reload --port 8000
```

2. Example curl request:

```bash
curl -X POST "http://127.0.0.1:8000/caption" -F "file=@/path/to/image.jpg" -F "beam_size=3"
```

## Run Web UI (Gradio)

1. Start the API then in another terminal run:

```bash
python web/gradio_app.py
```

The Gradio UI will open locally and call the API.

## Docker

Build and run the image (example):

```bash
docker build -t imgcap:latest .
docker run -p 8000:8000 imgcap:latest
```

## Notes & Tips
- For fast experiments use Flickr8k/30k or a small CSV. Full COCO requires >50GB.
- Use GPU (CUDA) for training. If running into CUDA mismatch, verify drivers and torch/cuda versions.
- To speed up inference in production, export model to ONNX and serve with ONNX Runtime.
