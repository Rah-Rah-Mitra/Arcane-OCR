# User Guide

Complete usage instructions for Arcane OCR.

## Prerequisites

- Raspberry Pi with Hailo-8L NPU module
- Python 3.11+
- Hailo runtime packages installed (see [Pipeline Setup](PIPELINE_SETUP.md))
- OCR HEF models downloaded to `models/hailo8l/`

## Basic Usage

All commands go through the `run_ocr.sh` wrapper, which activates the virtual environment and runs the pipeline:

```bash
./scripts/run_ocr.sh --input <path> --output-dir <path> [options]
```

### Process a single image

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ocr_sample_image.png \
  --output-dir ./output/image_run
```

### Process a PDF

```bash
./scripts/run_ocr.sh \
  --input ./public/samples/contents_page_sample.pdf \
  --output-dir ./output/pdf_run
```

### Process a directory of files

```bash
./scripts/run_ocr.sh \
  --input ./public/images/ \
  --output-dir ./output/batch_run
```

### Limit PDF pages

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/partial \
  --max-pages 3
```

## Recommended Profiles

### A4-Optimized (best for PDF documents)

Renders PDFs at standard DPI and tiles at detector-native dimensions (544x960). No downsampling loss.

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/a4_run \
  --a4-mode \
  --pdf-dpi 200 \
  --tile-overlap-ratio 0.12 \
  --rec-batch-size 8
```

Performance: ~2.5s per A4 page, 6 tiles per page at 200 DPI.

### Fast (lowest latency)

For quick previews or large batch processing where speed matters more than accuracy.

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/fast_run \
  --pdf-scale 2.0 \
  --tile-size 1280 \
  --tile-overlap-ratio 0.08 \
  --max-tiles-per-page 6 \
  --rec-batch-size 12 \
  --nms-iou-threshold 0.5
```

### Balanced (general purpose)

Good trade-off between speed and accuracy for mixed document types.

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/balanced_run \
  --pdf-scale 2.5 \
  --tile-size 1024 \
  --tile-overlap-ratio 0.12 \
  --max-tiles-per-page 9 \
  --rec-batch-size 8 \
  --nms-iou-threshold 0.5
```

### Max Accuracy (dense documents)

For tables of contents, dense tables, or fine print. More tiles, tighter overlap, stricter NMS.

```bash
./scripts/run_ocr.sh \
  --input document.pdf \
  --output-dir ./output/maxacc_run \
  --pdf-scale 3.0 \
  --tile-size 960 \
  --tile-overlap-ratio 0.15 \
  --max-tiles-per-page 0 \
  --rec-batch-size 4 \
  --nms-iou-threshold 0.45
```

## CLI Reference

### Input/Output

| Flag | Default | Description |
|------|---------|-------------|
| `--input`, `-i` | (required) | Input image, PDF, or directory |
| `--output-dir` | `./output` | Directory for output files |
| `--config` | `./config/ocr_config.yaml` | Config YAML path |
| `--max-pages` | `0` (all) | Limit number of PDF pages processed |

### PDF Rendering

| Flag | Default | Description |
|------|---------|-------------|
| `--a4-mode` | off | A4-optimized: render at DPI, tile at detector native 544x960 |
| `--pdf-dpi` | `200` | PDF render DPI (150=fast, 200=balanced, 300=high-res) |
| `--pdf-scale` | `2.0` | PDF render scale (used when `--a4-mode` is off) |

### Tiling

| Flag | Default | Description |
|------|---------|-------------|
| `--tile-size` | `1024` | Base tile size in pixels (non-A4 mode) |
| `--tile-overlap-ratio` | `0.12` | Overlap between adjacent tiles (0.0-0.95) |
| `--max-tiles-per-page` | `9` | Max tiles per page; auto-increases tile size. 0=unlimited |
| `--det-tile-height` | `544` | Detector tile height (A4 mode) |
| `--det-tile-width` | `960` | Detector tile width (A4 mode) |
| `--disable-tiling` | off | Process whole page without tiling |

### Edge Box Fusion

| Flag | Default | Description |
|------|---------|-------------|
| `--edge-threshold` | `0.03` | Edge band as fraction of tile size (0.03 = 3%) |
| `--fusion-threshold` | `0.5` | 1D IoU threshold for merging split detections |
| `--disable-edge-fusion` | off | Disable edge box fusion entirely |

### Local+Global Strategy

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-local-global` | off | Enable local+global tiling strategy |
| `--large-object-threshold` | `0.005` | Area ratio threshold; objects above this come from global view |

### Inference

| Flag | Default | Description |
|------|---------|-------------|
| `--det-hef` | from config | Override detector HEF path |
| `--rec-hef` | from config | Override recognizer HEF path |
| `--rec-batch-size` | `8` | Recognition crops per inference call |
| `--min-box-area` | `24` | Skip detections smaller than this area (px^2) |
| `--det-priority` | `0` | Scheduler priority for detector |
| `--rec-priority` | `1` | Scheduler priority for recognizer |
| `--group-id` | `SHARED` | Hailo VDevice group ID |
| `--tile-workers` | `1` | Parallel tile inference threads (1=sequential) |
| `--nms-iou-threshold` | `0.5` | IoU threshold for NMS duplicate suppression |

### Text Correction

| Flag | Default | Description |
|------|---------|-------------|
| `--use-corrector` | from config | Enable SymSpell text correction |
| `--disable-corrector` | off | Disable text correction |

## Output Files

Each processed page generates three files:

### Annotated Image (`<page>_ocr.png`)

Side-by-side view: original image on the left, annotated text on the right with bounding boxes and recognized text overlaid.

### Structured JSON (`<page>_structured.json`)

```json
{
  "lines": [
    {
      "line_id": 1,
      "indent_level": 0,
      "x_anchor": 85,
      "text": "Contents",
      "entries": [
        {
          "tile_index": 0,
          "crop_index": 3,
          "bbox": [85, 42, 180, 28],
          "text": "Contents",
          "score": 0.95
        }
      ]
    }
  ],
  "entries": [...]
}
```

Fields:
- `line_id` — sequential line number (top to bottom)
- `indent_level` — nesting depth based on x-position clustering
- `x_anchor` — leftmost x-coordinate of the line
- `bbox` — `[x, y, width, height]` in page pixel coordinates
- `score` — recognition confidence (0.0-1.0)

### Structured Markdown (`<page>_structured.md`)

```markdown
# OCR Structure: page_01

- Contents
  - Preface vii
  - Acknowledgments xi
    - Introduction 1
```

Indentation reflects spatial layout of the original document.

### Timing Report (`timing_report.json`)

Generated once per run in the output directory:

```json
{
  "total_seconds": 16.0,
  "items_processed": 7,
  "details": [
    {
      "item": "document_page_01",
      "elapsed_seconds": 1.87,
      "text_regions": 67,
      "tiles": 6,
      "tile_width": 960,
      "tile_height": 544,
      "render_mode": "a4-native",
      "tile_overlap_ratio": 0.12,
      "rec_batch_size": 8,
      "min_box_area": 24,
      "output": "..._ocr.png",
      "structured_json": "..._structured.json",
      "structured_markdown": "..._structured.md"
    }
  ]
}
```

## PDF DPI Selection

A4 page size: 210mm x 297mm

| DPI | Pixel Dimensions | Tiles (12% overlap) | Use Case |
|-----|-----------------|---------------------|----------|
| 150 | 1240 x 1754 | ~4 | Fast processing |
| 200 | 1653 x 2339 | ~6 | Balanced (recommended) |
| 250 | 2062 x 2924 | ~9 | Higher detail |
| 300 | 2480 x 3508 | ~12 | Maximum detail |

## Edge Box Fusion Explained

When tiling a page, text lines that cross tile boundaries are detected as fragments in both adjacent tiles. Without fusion, this causes duplicated or garbled text like "Contents Contents".

The fusion algorithm:

1. For each detection, checks if its bounding box falls within the `edge_threshold` band of an interior tile edge
2. Edge detections are paired using `max(iou_x, iou_y)` — a high 1D IoU indicates the same text line split across tiles
3. Pairs above `fusion_threshold` are merged: bounding boxes are unioned, the higher-confidence text is kept

Tuning:
- Increase `--edge-threshold` (e.g. 0.05) if you see duplicated text at tile borders
- Decrease `--fusion-threshold` (e.g. 0.3) to fuse more aggressively
- Use `--disable-edge-fusion` to turn off entirely for comparison

## Local+Global Strategy Explained

Some documents have both small text (body, page numbers) and large elements (chapter titles, headers). Tiles give high resolution for small text but can fragment large objects. The full image sees large objects clearly but misses small text detail.

With `--enable-local-global`:

1. Tile inference runs normally, producing all detections
2. Full-image inference runs on the entire page (downscaled to detector input)
3. **Small objects** (area < threshold * page_area) are kept from tile results
4. **Large objects** (area >= threshold * page_area) are kept from global results
5. Both sets are combined, then NMS removes remaining duplicates

Tune `--large-object-threshold` based on your document. Default 0.005 means objects covering >= 0.5% of page area are considered "large".

## NPU Monitoring

Verify that models run on the Hailo NPU:

```bash
# Terminal 1: start monitor
hailo monitor

# Terminal 2: run OCR
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output/test
```

The `run_ocr.sh` script sets `HAILO_MONITOR=1` by default. To disable:

```bash
HAILO_MONITOR=0 ./scripts/run_ocr.sh --input image.png --output-dir ./output/test
```

## Troubleshooting

### "Missing HEF files"

Run `./scripts/download_models.sh` or pass `--det-hef` and `--rec-hef` manually.

### "libhailort.so" errors

```bash
./scripts/install_hailo_runtime.sh
ldconfig -p | grep libhailort
```

### HEF compatibility errors

Ensure HEF models are compiled for HAILO8L (not HAILO8). Check `models/hailo8l/`.

### Text duplication in output

Edge box fusion should handle this. If you still see duplicates:
- Increase `--edge-threshold` to 0.05
- Decrease `--fusion-threshold` to 0.3
- Check if `--disable-edge-fusion` was accidentally set

### Poor recognition on small text

- Increase `--pdf-dpi` to 250 or 300
- Use `--a4-mode` for consistent rendering
- Decrease `--min-box-area` to avoid filtering legitimate small detections

### Slow performance

- Use `--a4-mode` with default 200 DPI
- Increase `--rec-batch-size` to 12
- Try `--tile-workers 2` for parallel tile processing
- Set `--max-tiles-per-page` to limit tile count
