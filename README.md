<div align="center">

# 🛰️ TerraVision

### AI-Powered Land-Cover Mapping & Change Detection

Deep learning (U-Net) + GIS statistics + a full-featured Flask web app — packaged as a portfolio project.

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-3.0-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed?logo=docker&logoColor=white)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-pytest-0a9edc?logo=pytest&logoColor=white)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-green)](#-license)

**No training required to try it** — ships with a documented heuristic classifier so the full
pipeline runs end-to-end on day one. Drop in a trained checkpoint later and it switches to real
CNN inference automatically.

</div>

```
Satellite Image  →  U-Net CNN (or heuristic fallback)  →  Segmentation  →  Statistics & Reports
```

**Classes:** 🏙️ Urban · 🌲 Forest · 💧 Water · 🌾 Agriculture · 🏜️ Bare land

<br>

## 📚 Table of contents

<table>
<tr><td width="50%" valign="top">

- [✨ Features](#-features)
- [⚡ Quickstart — no Docker](#-quickstart--no-docker-recommended)
- [🐳 Quickstart — Docker](#-quickstart--docker)
- [📁 Project structure](#-project-structure)
- [🧠 How it works](#-how-it-works)

</td><td width="50%" valign="top">

- [🔌 API reference](#-api-reference)
- [🎓 Training the real CNN](#-training-the-real-cnn)
- [📊 Evaluating a trained model](#-evaluating-a-trained-model)
- [🧪 Testing](#-testing)
- [🛠️ Troubleshooting](#️-troubleshooting)

</td></tr>
</table>

<br>

## ✨ Features

<table>
<tr>
<td width="33%" valign="top">

### 🗺️ Segmentation & analysis
- Tiled, full-resolution U-Net inference (no downscaling large scenes)
- Documented heuristic fallback — works with zero training
- Vegetation health index (proxy-NDVI), per-image normalized
- Per-pixel confidence map
- FRAGSTATS-style landscape fragmentation metrics
- Real-world area via optional GSD input
- CNN-vs-heuristic comparison mode

</td>
<td width="33%" valign="top">

### 🔄 Change detection
- Before/after % cover per class
- Full 5×5 transition matrix
- **Urban Expansion Rate** summary index
- **Vegetation Loss Rate** summary index
- Side-by-side before/after maps

</td>
<td width="33%" valign="top">

### 📄 Reporting & UI
- One-click PDF reports (ReportLab)
- CSV / JSON statistics export
- SQLite-backed run history
- Day/night theme toggle
- Drag-and-drop upload
- Before/after blend slider
- Click-to-isolate legend

</td>
</tr>
</table>

<details>
<summary><b>🧩 Engineering details</b></summary>
<br>

| Component | What it does |
|---|---|
| `model/evaluate.py` | Confusion matrix + per-class precision/recall/F1/IoU + heatmap |
| `tests/test_inference.py` | pytest suite — classification, stats, landscape metrics, change detection (heuristic-only, CI-safe) |
| `Dockerfile` + `docker-compose.yml` | One-command containerized deployment with persistent volumes for checkpoints and history |

</details>

<br>

## ⚡ Quickstart — no Docker (recommended)

The fastest way to see it running, with live-reload on every save.

```bash
cd landcover-classifier
pip install -r requirements.txt     # includes PyTorch — first install takes a few minutes
python app.py
```

Open **http://localhost:5000** 🎉

> 🪟 **Windows:** if `python` isn't recognized, use `py -m pip install -r requirements.txt` and `py app.py`.

Since no checkpoint exists yet, the app boots in **heuristic mode** — a documented, rule-based
classifier using RGB spectral cues. It has no learning capacity, but the entire pipeline (upload →
segmentation → stats → vegetation index → confidence map → landscape metrics → change detection)
works end-to-end out of the box. `app.py` runs with `debug=True`, so editing
`templates/index.html` or any `.py` file and refreshing the browser shows changes instantly — no
restart needed.

<br>

## 🐳 Quickstart — Docker

```bash
docker compose up --build
```

Open **http://localhost:5000**. `checkpoints/` and `data/` are mounted as volumes — drop a
trained `best.pt` into `checkpoints/` and restart the container to switch to CNN mode. Analysis
history persists across restarts in a named volume.

<details>
<summary>💡 Want live-reload inside Docker too?</summary>
<br>

By default only `checkpoints/`, `data/`, and the history volume are mounted — source code is
baked into the image at build time. Either:
- re-run `docker compose up --build` after every change, **or**
- add a source mount to `docker-compose.yml`:
  ```yaml
  volumes:
    - .:/app
    - ./checkpoints:/app/checkpoints
    - ./data:/app/data
    - terra-history:/app/history-data
  ```

</details>

<br>

## 📁 Project structure

```
landcover-classifier/
├── app.py                    Flask backend — UI, inference, PDF export, history API
├── templates/
│   └── index.html            Web UI: classify · change detection · history · methods
├── model/
│   ├── unet.py                U-Net architecture (PyTorch)
│   ├── dataset.py              Dataset loader + augmentation + class-weight helper
│   ├── train.py                 Training loop (Dice + CrossEntropy, mIoU, checkpointing)
│   ├── evaluate.py               Confusion matrix + per-class P/R/F1/IoU report
│   ├── inference.py                CNN (tiled) + heuristic engines, vegetation index,
│   │                                confidence map, landscape metrics
│   ├── report.py                    PDF report generation (ReportLab)
│   ├── history.py                    SQLite log of past analyses
│   └── prepare_masks.py               RGB label images → class-index masks
├── tests/
│   └── test_inference.py     pytest suite (heuristic path, no GPU/dataset needed)
├── checkpoints/               Created after training — best.pt / last.pt live here
├── data/                       Training data (train/ + val/, images + masks)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

<br>

## 🧠 How it works

| Output | How it's computed |
|---|---|
| **Segmentation** | Trained U-Net CNN (if `checkpoints/best.pt` exists) or a documented RGB spectral-index heuristic — assigns each pixel to one of the 5 classes. |
| **Vegetation index** | Proxy-NDVI (Excess Green: `2G − R − B`), stretched per-image to its actual min/max, rendered brown → yellow → green. Swap in a real NIR band for true NDVI = `(NIR−R)/(NIR+R)`. |
| **Confidence map** | CNN: `1 − normalized softmax entropy` per pixel. Heuristic: distance from the nearest decision boundary (vegetation / water / urban-vs-bare / brightness), stretched per-image to the full range. |
| **Landscape metrics** | Patch count, mean patch size, largest-patch dominance, edge density — FRAGSTATS-style fragmentation indicators from landscape ecology. |
| **Change-detection summary** | **Urban Expansion Rate** = % of scene newly converted to Urban. **Vegetation Loss Rate** = % of previously vegetated (Forest + Agriculture) area converted to something else. |
| **Large images (CNN mode)** | Overlapping sliding-window tiles, stitched at full resolution with majority voting in overlaps — instead of downscaling the whole scene into one 256×256 input. |
| **History** | Every run logged to `history.db` (SQLite) — survives refreshes and restarts. |
| **Reports** | "Export PDF report" builds a print-ready document server-side via ReportLab. |

<br>

## 🔌 API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/classify` | Segmentation + stats + vegetation index + confidence map + landscape metrics |
| `POST` | `/api/change-detection` | Before/after change stats + transition matrix |
| `POST` | `/api/export-pdf` | PDF report for a single-image classification |
| `POST` | `/api/export-change-pdf` | PDF report for a change-detection run |
| `GET` | `/api/history?limit=20` | Recent analyses |
| `DELETE` | `/api/history` | Clear all history |

<details>
<summary><b>Form fields (multipart/form-data)</b></summary>
<br>

| Field | Used by | Description |
|---|---|---|
| `image` | `/api/classify`, `/api/export-pdf` | The image to classify |
| `image_before`, `image_after` | `/api/change-detection`, `/api/export-change-pdf` | The two images to compare |
| `gsd` | all analysis endpoints | Optional ground sample distance (m/pixel) — enables hectare/km² area reporting |
| `engine` | `/api/classify`, `/api/change-detection` | Optional override: `"cnn"` or `"heuristic"` (omit for auto) |

</details>

<br>

## 🎓 Training the real CNN

<table>
<tr><td width="8%" valign="top"><b>1</b></td><td>

**Get labeled data.** Recommended public datasets for 5-class land cover:
[LandCover.ai](https://landcover.ai.linuxpolska.com/) · [DeepGlobe](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) · or your own tiles from QGIS / Google Earth Engine.

</td></tr>
<tr><td valign="top"><b>2</b></td><td>

**Arrange your data:**
```
data/
  train/images/*.png   train/masks/*.png   (mask = single-channel, values 0-4)
  val/images/*.png     val/masks/*.png
```
RGB label images instead of class-index masks? Convert them first:
```bash
python model/prepare_masks.py --src data/train/labels_rgb --dst data/train/masks
```

</td></tr>
<tr><td valign="top"><b>3</b></td><td>

**Train:**
```bash
python model/train.py --data-root ./data --epochs 50 --batch-size 16 --lr 1e-4
```
Saves `checkpoints/best.pt` (highest val mIoU) and `checkpoints/last.pt`. Class weights are
computed automatically to counter land-cover class imbalance.

</td></tr>
<tr><td valign="top"><b>4</b></td><td>

**Restart the app** (`python app.py`). It auto-detects `checkpoints/best.pt` and switches to real
CNN inference — the header badge flips from *"Heuristic baseline"* to *"Trained CNN (U-Net)"*.

</td></tr>
</table>

<br>

## 📊 Evaluating a trained model

```bash
python model/evaluate.py --data-root ./data --split val --checkpoint checkpoints/best.pt
```

Writes `checkpoints/eval_report.json` (accuracy, mean IoU, mean F1, per-class P/R/F1/IoU/support)
and a confusion-matrix heatmap PNG if matplotlib is installed.

<br>

## 🧪 Testing

```bash
pip install pytest
pytest tests/ -v
```

Runs against the heuristic classifier — no checkpoint, GPU, or dataset required. Safe for CI.

<br>

## 🛠️ Troubleshooting

<details>
<summary><b>🐳 <code>docker</code> is not recognized as a command</b></summary>
<br>

Docker Desktop isn't installed or isn't on your PATH. Install it from
[docker.com](https://www.docker.com/products/docker-desktop/) and restart your machine — or skip
Docker entirely and use the [no-Docker quickstart](#-quickstart--no-docker-recommended) instead.

</details>

<details>
<summary><b>✏️ I edited <code>templates/index.html</code> but nothing changed</b></summary>
<br>

1. **Hard-refresh** the browser: `Ctrl+Shift+R`.
2. Check the terminal running `python app.py` — saving a template should print
   `* Detected change in ... reloading`. If it doesn't appear, confirm `debug=True` is still set
   in `app.py`.
3. **Using Docker without a source volume mount?** Your edit never reaches the container — add
   `- .:/app` to `docker-compose.yml`, or rebuild with `docker compose up --build`.
4. Check for a stray duplicate file: `Get-ChildItem -Recurse -Filter index.html` (PowerShell).

</details>

<details>
<summary><b>🪟 <code>python</code> is not recognized (Windows)</b></summary>
<br>

Use `py` instead: `py -m pip install -r requirements.txt` and `py app.py`.

</details>

<details>
<summary><b>🐌 <code>pip install</code> is slow or seems stuck</b></summary>
<br>

`requirements.txt` includes `torch`, `torchvision`, `scipy`, and `matplotlib` — a multi-gigabyte
install, mostly PyTorch. This is normal and can take several minutes; no GPU is required to
install or to run the heuristic fallback.

</details>

<details>
<summary><b>🟡 Confidence map or vegetation index looks flat / solid-colored</b></summary>
<br>

Fixed as of the current `model/inference.py` — both outputs were previously computed against a
fixed, unrealistic value range instead of the image's actual min/max, collapsing almost every
pixel to the same brightness or color. Both are now normalized per-image. If you still see this,
confirm you're running the latest `model/inference.py`.

</details>

<br>

## 🏗️ Architecture notes

- **U-Net** — encoder-decoder with skip connections, 4 downsampling stages, bilinear upsampling by
  default (`bilinear=False` in `unet.py` for learned transposed-conv upsampling instead).
- **Loss:** CrossEntropy + Dice — handles class imbalance and small, irregular regions (rivers,
  roads) better than CrossEntropy alone.
- **Metric:** mean IoU (mIoU) across the 5 classes.
- **Multispectral-ready:** with a NIR band (e.g. Sentinel-2), set `in_channels=4` in `UNet(...)`
  and feed RGB+NIR for a true NDVI-like signal — substantially improves forest/agriculture
  separation over RGB-only imagery.

**Statistics & change detection**
- `stats_from_mask()` — % cover per class from pixel counts
- `change_detection()` — before/after % cover, per-class delta, and a full 5×5 transition matrix
  (e.g. Agriculture → Urban = urban sprawl into farmland) — the standard GIS land-cover-change
  output

<br>

## 🚀 Extending this project

- [ ] Swap Flask for FastAPI + async inference for higher throughput
- [ ] Add `rasterio`/`GDAL` support for GeoTIFF tiles with preserved CRS/geolocation
- [ ] Add a confusion matrix + per-class precision/recall to `train.py`'s training loop itself
- [ ] Export the trained model via `torch.jit.trace` or ONNX for lighter, faster deployment

<br>

## 📄 License

This project is licensed under the [MIT License](LICENSE).

<br>

<div align="center">

Built with 🛰️ + 🐍 + ☕

</div>
