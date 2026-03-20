# OCR Pipeline Optimization Guide

Tiling strategies, benchmarks, and tuning for Arcane OCR.

## The Problem: Resolution Loss in Tiling

The detector model expects **544x960** input. Large pages (A4 at 1600-3500px) must be tiled. Previous square tiling caused downsampling loss:

```
Old: PDF at scale 2.5 → Tile 1024x1024 → Shrink to 544x960 → Detail lost
New: PDF at 200 DPI  → Tile 960x544 (native) → No shrinking → Full detail
```

## Model Input Dimensions

| Model | Input Shape | Output Shape | Purpose |
|-------|------------|---------------|---------|
| `ocr_det.hef` | 544x960x3 (HxWxC) | 544x960x1 (heatmap) | Text region detection |
| `ocr.hef` | 48x320x3 (HxWxC) | 1x40x97 (char sequence) | Character recognition |

## Tiling Strategies

### 1. Aspect-Ratio-Aware Tiling (default)

Uses `degirum_tools.generate_tiles_fixed_size()` to create a 2D grid matching the detector's native 960x544 (WxH) aspect ratio. Each tile is sent directly to the model without downsampling.

```
A4 at 200 DPI (1653x2339px):
  Horizontal: ⌈(1653 - 960) / (960 × 0.88)⌉ + 1 = 2-3 tiles
  Vertical:   ⌈(2339 - 544) / (544 × 0.88)⌉ + 1 = 2 tiles
  Total: ~6 tiles per page
```

Enabled by default. Previously used square tiles with `generate_sliding_windows()` (still available as fallback).

### 2. Edge Box Fusion (default)

Solves text duplication at tile boundaries:

```
Without fusion:  "Contents" "Contents" → "Contents Contents"
With fusion:     "Conte" + "ntents"   → "Contents" (merged, higher-confidence text kept)
```

Algorithm:
1. **Classify**: each detection as central (far from tile edges) or edge (within `edge_threshold` band of an interior tile boundary)
2. **Match**: edge detections using `max(iou_x, iou_y)` — split text fragments have high 1D IoU in the axis parallel to the tile split
3. **Fuse**: pairs above `fusion_threshold` are merged — bounding boxes are unioned, higher-confidence text is retained

Parameters:
- `--edge-threshold 0.03` — edge band width as fraction of tile dimension (3% = ~16px for 544px tile)
- `--fusion-threshold 0.5` — 1D IoU threshold for merging
- `--disable-edge-fusion` — turn off for comparison

### 3. Local+Global Strategy (opt-in)

Addresses the tile/full-image trade-off:

| Inference Mode | Sees Small Text Well | Sees Large Objects Well |
|---------------|---------------------|------------------------|
| Tiles only | Yes | No (fragmented) |
| Full image only | No (downscaled) | Yes |
| Local+Global | Yes (from tiles) | Yes (from full image) |

With `--enable-local-global`:
1. Tile inference produces all detections
2. Full-image inference runs on the entire page
3. Small objects (area < `large_object_threshold` x page_area) kept from tiles
4. Large objects (area >= threshold x page_area) kept from global
5. NMS removes remaining duplicates

### 4. Multi-threaded Tile Inference (opt-in)

Overlaps Python preprocessing (resize, pad, crop extraction) with NPU inference:

```
Sequential:  [preprocess tile 1][infer tile 1][preprocess tile 2][infer tile 2]...
Parallel:    [preprocess tile 1][infer tile 1]
                [preprocess tile 2][infer tile 2]
                   [preprocess tile 3][infer tile 3]...
```

The Hailo `wait_for_async_ready()` gate serializes NPU access, so more than 2-3 workers provides no additional benefit.

Enable with `--tile-workers 2`.

## OpenCV NMS

Replaced the O(n^2) Python greedy NMS loop with `cv2.dnn.NMSBoxes()`:

- Same input format: `[x, y, w, h]` bounding boxes + confidence scores
- Same output: list of kept indices
- Runs in C++, significantly faster for pages with hundreds of detections

## A4 DPI Reference

A4 page size: 210mm x 297mm

| DPI | Pixel Dimensions | Tiles (12% overlap, 960x544) | Speed vs 300 DPI |
|-----|-----------------|------------------------------|-------------------|
| 150 | 1240 x 1754 | ~4 | ~3x faster |
| 200 | 1653 x 2339 | ~6 | **2.6x faster** |
| 250 | 2062 x 2924 | ~9 | ~1.5x faster |
| 300 | 2480 x 3508 | ~12 | baseline |

## Performance Benchmarks

7-page PDF on Raspberry Pi + Hailo-8L:

| Configuration | Runtime | Tiles/Page | Regions | Notes |
|--------------|---------|------------|---------|-------|
| A4-Native 200 DPI | **16.0s** | 6 | ~870 | Recommended |
| A4-Native 200 DPI + edge fusion | **16.0s** | 6 | ~870 | Default (fusion overhead minimal) |
| Balanced (1024px tiles) | ~25s | 9 | ~900 | Legacy preset |
| Max Accuracy (960px, 15% overlap) | ~46s | 9+ | ~1000 | Dense documents |

Key observations:
- A4-native is fastest because detector-native tiles skip downsampling
- Edge fusion adds negligible overhead (~10ms per page) but eliminates text duplication
- Region count is comparable — fewer tiles doesn't mean fewer detections
- Quality parity or better due to zero shrinking artifacts

## Tuning Guide

### Recognizer Batch Size (`--rec-batch-size`)

The biggest speed lever. Groups text crops per inference call:

| Value | Effect |
|-------|--------|
| 12 | Fastest, may slightly reduce per-crop quality |
| 8 | Balanced (recommended) |
| 4 | More accurate per crop, slower |
| 1 | Reference quality, slowest |

### Tile Overlap (`--tile-overlap-ratio`)

| Value | Effect |
|-------|--------|
| 0.08 | Minimal overlap, fastest, risk of missing text at boundaries |
| 0.12 | Balanced (default), good with edge fusion |
| 0.15 | Conservative, maximum text preservation |

### NMS IoU Threshold (`--nms-iou-threshold`)

| Value | Effect |
|-------|--------|
| 0.5 | Standard (default), keeps more detections |
| 0.45 | Stricter, removes more overlapping boxes |
| 0.3 | Aggressive, may remove legitimate adjacent text |

### Minimum Box Area (`--min-box-area`)

Skip tiny noisy detections:

| Value | Effect |
|-------|--------|
| 24 | Default, filters noise while keeping small text |
| 16 | Keep more small detections |
| 48 | Skip more noise, may miss subscripts/superscripts |

### Scheduler Priority (`--det-priority`, `--rec-priority`)

Tunes NPU scheduling between detector and recognizer:

```bash
# Prioritize detection (bottlenecked on finding text)
--det-priority 1 --rec-priority 0

# Prioritize recognition (bottlenecked on reading text)
--det-priority 0 --rec-priority 1
```

### Adaptive Tile Budget (`--max-tiles-per-page`)

Automatically increases tile size to stay within budget:

| Value | Effect |
|-------|--------|
| 0 | Disabled (fixed tile size) |
| 6 | At most 6 tiles per page |
| 9 | Default, good for A4 at 200-250 DPI |
| 16 | Allows high tile count for large pages |

## Recommended Configurations

### General Documents (reports, letters, forms)

```bash
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output \
  --a4-mode --pdf-dpi 200 --tile-overlap-ratio 0.12 --rec-batch-size 8
```

### Dense Tables / Tables of Contents

```bash
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output \
  --a4-mode --pdf-dpi 250 --tile-overlap-ratio 0.12 --rec-batch-size 6
```

### Documents with Large Headers

```bash
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output \
  --a4-mode --enable-local-global --large-object-threshold 0.005
```

### Maximum Quality (reference runs)

```bash
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output \
  --a4-mode --pdf-dpi 300 --tile-overlap-ratio 0.15 \
  --rec-batch-size 4 --nms-iou-threshold 0.45
```

### Maximum Speed (batch processing)

```bash
./scripts/run_ocr.sh --input doc.pdf --output-dir ./output \
  --a4-mode --pdf-dpi 150 --rec-batch-size 12 --tile-workers 2
```
