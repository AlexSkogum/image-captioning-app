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

### Using a Kaggle dataset (recommended workflow)

Do NOT commit large datasets to this repository. Instead, use the Kaggle CLI to download the dataset locally and create a small sample for submission/testing. The steps below explain the recommended reproducible workflow.

1. Install the Kaggle CLI in your project venv:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install kaggle
```

2. Place your `kaggle.json` token in `%USERPROFILE%\\.kaggle\\kaggle.json` (do NOT commit this file).

3. Download the dataset (example) and unzip into `data/`:

```powershell
# replace owner/dataset-name with the Kaggle dataset id
kaggle datasets download -d owner/dataset-name -p data/ --unzip
```

4. Create a small sample for quick tests (this repo includes helper scripts):

```powershell
# create a 100-image sample from COCO-style annotations
python scripts/create_sample_from_coco.py --annotations data/annotations/captions_train2017.json --images-dir data/train2017 --sample-size 100 --out-dir data/sample
```

5. The sample CSV at `data/sample/sample_captions.csv` can be used with the project's preprocessing and training commands.

6. In your submission, include a short note pointing graders to the Kaggle dataset link and the commands above so they can reproduce results. This keeps the GitHub repo small and dependency-free of large data.

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
