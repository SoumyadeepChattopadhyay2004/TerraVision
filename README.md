# Terra — Satellite Land-Cover Classification

Deep learning (U-Net) + GIS statistics + web app, packaged as a portfolio project.

```
Satellite Image → CNN / U-Net → Segmentation → Land-cover map → Statistics
```

Classes: **Urban · Forest · Water · Agriculture · Bare land**

## What's included

```
landcover-classifier/
├── app.py                  Flask backend (UI + inference + PDF export + history API)
├── templates/index.html    Web UI: classify, change detection, history, methods
├── model/
│   ├── unet.py              U-Net architecture (PyTorch)
│   ├── dataset.py           Dataset loader + augmentation + class-weight helper
│   ├── train.py             Training loop (Dice + CrossEntropy loss, mIoU, checkpointing)
│   ├── evaluate.py          Confusion matrix + per-class precision/recall/F1/IoU report
│   ├── inference.py         CNN (tiled, full-resolution) + heuristic fallback, vegetation
│   │                        index, confidence map, landscape fragmentation metrics
│   ├── report.py            PDF report generation (ReportLab)
│   ├── history.py           SQLite log of past analyses
│   └── prepare_masks.py     Converts RGB label images -> class-index masks
├── tests/test_inference.py  pytest suite (heuristic path, no GPU/dataset needed)
├── Dockerfile / docker-compose.yml   One-command deployment
├── requirements.txt
└── checkpoints/             (created after training; holds best.pt / last.pt)
```

## Full feature set

**Segmentation & analysis**
- U-Net CNN segmentation with **tiled, full-resolution inference** for large images (sliding-window with overlap voting, instead of squashing everything to 256×256)
- Documented heuristic fallback so the app works before you've trained anything
- Vegetation health index (proxy-NDVI), per-pixel model confidence map
- Landscape fragmentation metrics (patch count, mean patch size, largest-patch dominance, edge density) — FRAGSTATS-style, standard in landscape ecology
- Real-world area (hectares/km²) via ground sample distance input
- CNN-vs-heuristic side-by-side comparison mode

**Change detection**
- Before/after % cover, full 5×5 transition matrix, Urban Expansion Rate and Vegetation Loss Rate summary indices

**Reporting & persistence**
- One-click PDF report export (maps, stats table, vegetation index, landscape metrics) via `model/report.py`
- CSV / JSON statistics export
- Every run is logged to a local SQLite database (`model/history.py`) — a History tab lets you revisit past analyses after a refresh or restart

**Engineering**
- `model/evaluate.py` — confusion matrix + per-class precision/recall/F1/IoU + heatmap, the report an academic write-up actually needs (training-loop mIoU alone hides which classes get confused)
- `tests/test_inference.py` — pytest suite covering classification, stats, landscape metrics, and change detection
- `Dockerfile` + `docker-compose.yml` — one-command deployment with persistent volumes for checkpoints and history

**UI**
- Day/night theme toggle, drag-and-drop upload, before/after blend slider, click-to-isolate legend, interactive charts

## Quickstart (works immediately, no training required)

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Since no trained checkpoint exists yet, the app
automatically runs in **heuristic mode**: a documented, rule-based classifier
using RGB spectral cues (excess-green vegetation index, blue/darkness for
water, brightness + saturation for urban vs. bare land). It has no learning
capacity, but it means the full pipeline — upload, segmentation map overlay,
per-class statistics, and change detection between two dates — works end to
end out of the box. This is your fallback / demo mode, not the final model.

## Training the real CNN

1. **Get labeled data.** Recommended public datasets for 5-class land cover:
   - [LandCover.ai](https://landcover.ai.linuxpolska.com/) (buildings, woodland, water, roads — remap to your 5 classes)
   - [DeepGlobe Land Cover Classification](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) (urban, agriculture, rangeland, forest, water, barren — very close match)
   - Your own labeled tiles from QGIS / Google Earth Engine exports

2. **Arrange data** in this layout:
   ```
   data/
     train/images/*.png   train/masks/*.png   (mask = single-channel, pixel value 0-4)
     val/images/*.png     val/masks/*.png
   ```
   If your dataset ships masks as solid-color RGB images instead of class
   indices, convert them first:
   ```bash
   python model/prepare_masks.py --src data/train/labels_rgb --dst data/train/masks
   ```
   (edit `SOURCE_COLOR_MAP` in that file to match your dataset's actual palette)

3. **Train:**
   ```bash
   python model/train.py --data-root ./data --epochs 50 --batch-size 16 --lr 1e-4
   ```
   This saves `checkpoints/best.pt` (highest validation mIoU) and `checkpoints/last.pt`.
   Class weights are computed automatically to counter land-cover class imbalance
   (water and urban tiles are typically much rarer than forest/agriculture).

4. **Restart the app** (`python app.py`). It auto-detects `checkpoints/best.pt`
   and switches to real CNN inference — no code changes needed.

## Evaluating a trained model

```bash
python model/evaluate.py --data-root ./data --split val --checkpoint checkpoints/best.pt
```
Writes `checkpoints/eval_report.json` (overall accuracy, mean IoU, mean F1,
and per-class precision/recall/F1/IoU/support) and, if matplotlib is
installed, a confusion-matrix heatmap PNG next to it.

## Testing

```bash
pip install pytest
pytest tests/ -v
```
The suite runs against the heuristic classifier, so it needs no trained
checkpoint, GPU, or dataset — safe to run in CI.

## Docker deployment

```bash
docker compose up --build
```
Open http://localhost:5000. `checkpoints/` and `data/` are mounted as
volumes, so dropping a trained `best.pt` into `checkpoints/` on the host and
restarting the container switches it to CNN mode; analysis history persists
in a named volume across restarts.

## Architecture notes

- **U-Net**, standard encoder-decoder with skip connections, 4 downsampling
  stages, bilinear upsampling by default (set `bilinear=False` in `unet.py`
  for learned transposed-conv upsampling instead).
- **Loss:** CrossEntropy + Dice, which handles class imbalance and small,
  irregularly shaped regions (e.g. rivers, roads) better than CrossEntropy alone.
- **Metric:** mean IoU (mIoU) across the 5 classes, the standard segmentation metric.
- **Multispectral-ready:** if your imagery has a NIR band (e.g. Sentinel-2),
  set `in_channels=4` in `UNet(...)` and feed RGB+NIR — this lets the network
  learn a true NDVI-like signal, which substantially improves
  forest/agriculture separation versus RGB-only imagery.

## Statistics & change detection

- `stats_from_mask()` — % cover per class from pixel counts.
- `change_detection()` — before/after % cover, per-class delta, and a full
  5×5 transition matrix (e.g. how many pixels went from Agriculture → Urban,
  i.e. urban sprawl into farmland) — the standard GIS land-cover-change output.

## Extending this project

- Swap the Flask API for FastAPI + async inference for higher throughput.
- Add real geospatial I/O with `rasterio`/`GDAL` to accept GeoTIFF tiles and
  preserve CRS/geolocation, so statistics can be reported in km² instead of %.
- Add a confusion matrix + per-class precision/recall to `train.py` for a
  fuller evaluation report.
- Serve the trained model via `torch.jit.trace` or ONNX export for faster,
  dependency-light deployment.
