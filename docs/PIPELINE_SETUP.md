# Full Pipeline Setup (OCR-Only)

This document describes the complete setup flow for this OCR-only project.

## Required Local Packages

Expected in `setup-files/`:

- `hailort-pcie-driver_4.23.0_all.deb`
- `hailort_4.23.0_arm64.deb`
- `hailo-tappas-core_5.2.0_arm64.deb`
- `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`
- `hailo_tappas_core_python_binding-5.2.0-py3-none-any.whl`

## Setup Sequence

1. Recreate project venv:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

2. Install runtime components:

```bash
source .venv/bin/activate
./scripts/install_hailo_runtime.sh
```

3. Download OCR models:

```bash
./scripts/download_models.sh
```

4. Validate package state:

```bash
./scripts/check_installed_packages.sh
ldconfig -p | grep -i libhailort
python -m pip show hailort hailo-tappas-core-python-binding
```

## Run Examples

### Image OCR

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ocr_sample_image.png \
  --output-dir ./output/image_run
```

### PDF OCR

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_run
```

### High-Precision PDF OCR (Dense Pages)

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_tiled_run \
  --tile-size 1024 \
  --tile-overlap-ratio 0.12
```

This mode uses overlapping sliding windows, maps tile detections back to page coordinates, and applies NMS to remove duplicates from overlap regions.

### A4-Optimized High-Throughput OCR (Recommended for PDFs)

A4-aware rendering with detector-native tiling eliminates resolution loss:

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_a4native_run \
  --a4-mode \
  --pdf-dpi 200 \
  --tile-overlap-ratio 0.12 \
  --rec-batch-size 8
```

This mode applies three key optimizations:

- **Standard A4 DPI rendering**: PDF pages rendered at 200 DPI (standard for document OCR on A4 size).
- **Detector native tiling**: 544×960 tiles match detector input dimensions exactly, eliminating shrinking loss.
- **Batched recognizer inference**: Groups text crops for higher throughput.

Performance:

- 7-page PDF: 17.7 seconds total (874 regions)
- 6 tiles per page
- 2.6× faster than max-accuracy tiling mode
- Quality parity or better (no downsampling artifacts)

DPI selection:

- `--pdf-dpi 150`: Faster, lower detail (1240×1754 A4 pixels)
- `--pdf-dpi 200`: Balanced (1653×2339 A4 pixels) - **recommended**
- `--pdf-dpi 250`: Higher detail (2062×2924 A4 pixels)
- `--pdf-dpi 300`: Maximum detail (2480×3508 A4 pixels)

### Balanced High-Throughput OCR

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_balanced_run \
  --pdf-scale 2.5 \
  --tile-size 1024 \
  --tile-overlap-ratio 0.12 \
  --max-tiles-per-page 9 \
  --rec-batch-size 8
```

This mode applies two key optimizations:

- Adaptive tile budgeting: avoids unnecessary tile count on high-resolution pages.
- Batched recognizer inference: groups text crops into larger inference calls to reduce per-call overhead.

Useful tuning flags:

- `--max-tiles-per-page 0`: disable adaptation (fixed tile size).
- `--min-box-area`: skip tiny detections that usually become OCR noise.
- `--det-priority` / `--rec-priority`: scheduler priority between detection and recognition.

## Output Artifacts

For each run:

- Annotated OCR output images in output directory
- `timing_report.json` with per-input timing
- `<page>_structured.json` with line-level hierarchy and global boxes
- `<page>_structured.md` with markdown hierarchy output

## Troubleshooting

- If `libhailort.so` errors appear:
  - re-run `./scripts/install_hailo_runtime.sh`
  - verify `ldconfig -p | grep libhailort`
- If HEF compatibility fails:
  - ensure HEFs come from `models/hailo8l/`
  - avoid mixing HAILO8 HEFs with HAILO8L device
