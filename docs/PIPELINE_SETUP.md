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
