# Arcane OCR (Standalone)

Standalone OCR-only project for Raspberry Pi + Hailo-8L.

This project contains only the OCR pipeline components (image + PDF) and excludes unrelated example apps.

## What Is Included

- Standalone OCR pipeline code under `src/arcane_ocr`
- Local runtime package artifacts under `setup-files`
- Model download script for Hailo8L OCR HEFs
- Clean sample assets in `public/images` and `public/samples`
- Runtime + venv setup scripts in `scripts`
- Per-item and total OCR timing report output

## Project Layout

- `src/arcane_ocr/pipeline.py`: OCR pipeline entrypoint (image/PDF)
- `src/arcane_ocr/hailo_inference.py`: Minimal Hailo inference wrapper
- `src/arcane_ocr/ocr_utils.py`: OCR decode/postprocess/visualization helpers
- `src/arcane_ocr/db_postprocess.py`: DB detector postprocess implementation
- `setup-files/`: Local .deb/.whl runtime install files
- `models/hailo8l/`: OCR HEF models
- `public/images/`: Image examples
- `public/samples/`: PDF sample and correction dictionary
- `output/`: OCR outputs and timing report

## Quick Start

### 1) Recreate virtual environment

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

### 2) Install runtime packages (system + active venv)

```bash
source .venv/bin/activate
./scripts/install_hailo_runtime.sh
```

### 3) Download OCR models

```bash
./scripts/download_models.sh
```

### 4) Verify runtime packages

```bash
./scripts/check_installed_packages.sh
```

### 5) Run OCR on image

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ocr_sample_image.png \
  --output-dir ./output/image_run
```

### 6) Run OCR on PDF

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_run
```

### 7) High-Precision PDF OCR (Tiled)

For dense CV/table-of-contents pages, run sliding-window OCR with overlap:

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_tiled_run \
  --tile-size 1024 \
  --tile-overlap-ratio 0.12
```

Default overlap is 12% and can be tuned in the 10-15% range.

## Live Verification (NPU Activity)

Use this to verify models are running on the Hailo NPU in real time.

1. Start monitor in Terminal A:

```bash
hailo monitor
```

2. Run OCR in Terminal B:

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_run
```

`run_ocr.sh` exports `HAILO_MONITOR=1` by default.
If you need to disable telemetry for a run:

```bash
HAILO_MONITOR=0 ./scripts/run_ocr.sh --input ./public/images/ocr_sample_image.png
```

3. Confirm monitor updates while OCR is running.

Notes:

- HEF models are loaded to the device at runtime for each execution.
- This project does not permanently flash models onto the chip.

## Timing Feature

The pipeline prints runtime per item and writes a report:

- Console output: `[TIME] <item>: <seconds>s`
- JSON report: `<output-dir>/timing_report.json`

Example report fields:

- `total_seconds`
- `items_processed`
- `details[].elapsed_seconds`
- `details[].text_regions`
- `details[].output`
- `details[].tiles`
- `details[].tile_size`
- `details[].tile_overlap_ratio`
- `details[].structured_json`
- `details[].structured_markdown`

## Structured Page Output

Each processed page writes:

- `<page>_structured.json`: line groups, indent levels, global bounding boxes, text, and scores
- `<page>_structured.md`: markdown hierarchy aligned from page spatial layout

The pipeline performs:

1. Sliding-window tiling with overlap.
2. Tile-to-global coordinate reconstruction.
3. NMS-based duplicate suppression for overlapping regions.

## Configuration

Default config is in `config/ocr_config.yaml`:

- detector model path
- recognizer model path
- correction dictionary path
- corrector enabled/disabled

You can override paths at runtime:

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ocr_sample_image.png \
  --det-hef ./models/hailo8l/ocr_det.hef \
  --rec-hef ./models/hailo8l/ocr.hef
```

## Notes

- Use Hailo8L-compatible HEFs for Hailo8L devices.
- The project supports image (`.png`, `.jpg`, `.jpeg`, `.bmp`) and PDF inputs.
- PDF pages are rendered and processed page-by-page.
- Assets in `public/` are already organized for image and PDF examples.
