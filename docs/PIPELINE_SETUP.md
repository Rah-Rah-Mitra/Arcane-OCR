# Full Pipeline Setup

Complete setup flow for Arcane OCR on Raspberry Pi + Hailo-8L.

## Required Local Packages

Expected in `setup-files/`:

- `hailort-pcie-driver_4.23.0_all.deb`
- `hailort_4.23.0_arm64.deb`
- `hailo-tappas-core_5.2.0_arm64.deb`
- `hailort-4.23.0-cp311-cp311-linux_aarch64.whl`
- `hailo_tappas_core_python_binding-5.2.0-py3-none-any.whl`

## Setup Sequence

### 1. Create virtual environment

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

### 2. Install Hailo runtime

```bash
source .venv/bin/activate
./scripts/install_hailo_runtime.sh
```

### 3. Download OCR models

```bash
./scripts/download_models.sh
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Validate installation

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

### PDF OCR (A4-Optimized, recommended)

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_a4_run \
  --a4-mode \
  --pdf-dpi 200 \
  --tile-overlap-ratio 0.12 \
  --rec-batch-size 8
```

This uses aspect-ratio-aware tiling at detector-native 544x960 dimensions with edge box fusion enabled by default.

### PDF OCR with edge fusion disabled (comparison)

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_no_fusion \
  --a4-mode \
  --disable-edge-fusion
```

### PDF OCR with local+global strategy

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_localglobal \
  --a4-mode \
  --enable-local-global \
  --large-object-threshold 0.005
```

### PDF OCR with parallel tile inference

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_parallel \
  --a4-mode \
  --tile-workers 2
```

### High-Precision PDF OCR (dense pages)

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_dense \
  --a4-mode \
  --pdf-dpi 300 \
  --tile-overlap-ratio 0.15 \
  --rec-batch-size 4 \
  --nms-iou-threshold 0.45
```

### Balanced High-Throughput OCR

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_balanced \
  --pdf-scale 2.5 \
  --tile-size 1024 \
  --tile-overlap-ratio 0.12 \
  --max-tiles-per-page 9 \
  --rec-batch-size 8
```

### Process without tiling (single-pass)

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ocr_sample_image.png \
  --output-dir ./output/no_tile \
  --disable-tiling
```

## Output Artifacts

Each run generates:

- `<page>_ocr.png` — annotated image with detected regions and recognized text
- `<page>_structured.json` — structured data with bounding boxes, confidence, and text
- `<page>_structured.md` — markdown hierarchy preserving indentation
- `timing_report.json` — per-page timing, tile count, region count, and settings used

## Configuration

Default settings in `config/ocr_config.yaml`:

```yaml
models:
  det_hef: ./models/hailo8l/ocr_det.hef
  rec_hef: ./models/hailo8l/ocr.hef
runtime:
  use_corrector: true
  dictionary_path: ./public/samples/frequency_dictionary_en_82_765.txt
```

All config values can be overridden via CLI flags.

## Troubleshooting

### `libhailort.so` errors

Re-run `./scripts/install_hailo_runtime.sh` and verify:

```bash
ldconfig -p | grep libhailort
```

### HEF compatibility failures

Ensure HEFs are from `models/hailo8l/` (compiled for HAILO8L, not HAILO8).

### `ModuleNotFoundError: degirum_tools`

Install with: `pip install degirum-tools`

### Text duplication in output

This is handled by edge box fusion (enabled by default). If you still see duplicates:

- Increase `--edge-threshold` to 0.05
- Decrease `--fusion-threshold` to 0.3

### Slow performance

- Use `--a4-mode` (avoids unnecessary resolution)
- Increase `--rec-batch-size` to 12
- Try `--tile-workers 2`
- Set `--max-tiles-per-page` to limit tile count
