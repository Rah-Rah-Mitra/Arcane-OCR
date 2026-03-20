# Developer Guide

Architecture, code structure, and extension points for Arcane OCR.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        pipeline.py                            │
│  CLI args → resolve inputs → tiling → inference → outputs     │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Tile Gen    │  │ Inference    │  │ Post-processing       │ │
│  │             │  │              │  │                       │ │
│  │ aspect_ratio│  │ det → crops  │  │ edge fusion           │ │
│  │ windows     │→ │ crops → text │→ │ NMS (cv2.dnn)         │ │
│  │             │  │              │  │ structure → outputs   │ │
│  └─────────────┘  └──────┬───────┘  └──────────────────────┘ │
│                          │                                    │
└──────────────────────────┼────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
  ┌───────▼──────┐  ┌──────▼──────┐  ┌─────▼──────────┐
  │hailo_inference│  │  ocr_utils  │  │db_postprocess  │
  │              │  │             │  │                │
  │HailoInfer   │  │preprocess   │  │DBPostProcess   │
  │VDevice mgmt │  │postprocess  │  │shapely+pyclipper│
  │async run    │  │visualize    │  │polygon ops     │
  └──────────────┘  └─────────────┘  └────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Hailo-8L   │
                    │  NPU HW    │
                    └─────────────┘
```

## Source Files

### `src/arcane_ocr/pipeline.py`

Main entrypoint. Contains:

- **CLI argument parsing** (`parse_args`) — all runtime configuration
- **Input resolution** (`resolve_input_items`) — images, PDFs, directories
- **Tiling strategies**:
  - `generate_sliding_windows()` — legacy square-tile fallback
  - `generate_aspect_ratio_windows()` — aspect-ratio-aware tiling via `degirum_tools.generate_tiles_fixed_size()`
  - `adapt_tile_size_for_budget()` — adaptive tile sizing for max-tiles constraint
- **NMS** (`nms_entries`) — uses `cv2.dnn.NMSBoxes` for O(n log n) C++ NMS
- **Edge box fusion**:
  - `_compute_grid_dims()` — infer grid shape from window list
  - `_max_1d_iou()` — 1D IoU metric for split detection matching
  - `classify_edge_entries()` — separate central vs edge detections
  - `fuse_edge_entries()` — merge split text fragments
- **Inference strategies**:
  - `infer_tile_entries()` — sequential per-tile detection + recognition
  - `run_local_global_inference()` — tiles + full-image merged by object size
  - `infer_tiles_parallel()` — thread-pooled tile processing
- **Output generation**:
  - `build_page_structure()` — line grouping, indent levels, spatial ordering
  - `write_structured_outputs()` — JSON and Markdown output files

### `src/arcane_ocr/hailo_inference.py`

Minimal async inference wrapper for `hailo_platform`:

- `HailoInfer` — loads a HEF model, configures batch size and priority, provides `run()` for async inference with callbacks
- `create_shared_vdevice()` — creates a shared VDevice with round-robin scheduling so detector and recognizer share the NPU

Key design: the `wait_for_async_ready()` gate serializes actual NPU access, making multi-threading safe but limiting parallel HW utilization to 1 model at a time.

### `src/arcane_ocr/ocr_utils.py`

OCR pre/postprocessing and visualization:

- `ocr_eval_postprocess()` — CTC decode: argmax → deduplicate → character lookup (97-char set)
- `default_preprocess()` — resize with letterbox padding to model dimensions
- `resize_with_padding()` — resize text crops to recognition model input (48x320)
- `det_postprocess()` — extract heatmap, run DB postprocess, warp-crop text regions
- `visualize_ocr_annotations()` — side-by-side original + annotated output
- `OcrCorrector` — SymSpell-based spelling correction

### `src/arcane_ocr/db_postprocess.py`

PaddlePaddle DB (Differentiable Binarization) postprocessor:

- `DBPostProcess` — binarize heatmap, find contours, unclip polygons via Vatti clipping (pyclipper), filter by score/area
- Uses `shapely` for polygon area and `pyclipper` for polygon expansion

### `src/arcane_ocr/combine_pages.py`

Multi-page hierarchy reconstruction:

- `combine_page_structures()` — merges per-page structured outputs across page boundaries using stack-based tree building
- `_build_hierarchy_from_lines()` — orchestrates page combining logic with header filtering
- `_parse_section_number()` — extracts leading numeric tokens for display formatting
- `_extract_trailing_page_num()` — detects both arabic and roman numeral page references
- `_is_page_header()` — filters repeated page headers using regex patterns
- `write_combined_outputs()` — generates both JSON (tree + flat array) and Markdown outputs

Key features: indent-level-based nesting, cross-page continuation via stack, section number parsing, part prefix detection, and page number extraction (arabic and roman).

## Inference Flow

```
1. Image/PDF → resolve_input_items()
   ├── Images: cv2.imread()
   └── PDFs: pypdfium2 render at DPI scale

2. Per page:
   a. Generate tile windows
      ├── A4 mode: generate_aspect_ratio_windows(det_tile_w, det_tile_h)
      └── Manual: adapt_tile_size → generate_aspect_ratio_windows()

   b. Run inference on tiles
      ├── Sequential: for each tile → infer_tile_entries()
      ├── Parallel: infer_tiles_parallel() via ThreadPoolExecutor
      └── Local+Global: run_local_global_inference()

   c. Per tile inference (infer_tile_entries):
      i.   default_preprocess() → letterbox to 544×960
      ii.  run_single_inference() on detector → heatmap
      iii. det_postprocess() → text crops + local bboxes
      iv.  resize_with_padding() each crop → 48×320
      v.   run_batch_inference() on recognizer → character sequences
      vi.  ocr_eval_postprocess() → (text, confidence)
      vii. Map local bbox → global coordinates

   d. Edge box fusion (if enabled):
      i.   _compute_grid_dims() → (n_rows, n_cols)
      ii.  classify_edge_entries() → central + edge lists
      iii. fuse_edge_entries() → merge split text at tile boundaries

   e. nms_entries() → cv2.dnn.NMSBoxes → deduplicated entries

   f. build_page_structure() → line groups with indent levels
   g. write_structured_outputs() → JSON + Markdown
   h. visualize_ocr_annotations() → annotated PNG
```

## Key Design Decisions

### Aspect-Ratio-Aware Tiling

Previous square tiles (e.g. 1024x1024) caused unnecessary stretching when fed to the 544x960 detector. Now tiles match the detector's native aspect ratio, sent directly without downsampling loss.

### Edge Box Fusion

Text lines crossing tile boundaries get detected as fragments in both adjacent tiles. The fusion algorithm:

1. Identifies detections near interior tile edges (within `edge_threshold` band)
2. Computes `max(iou_x, iou_y)` between edge detections — high 1D IoU indicates a split line
3. Merges pairs above `fusion_threshold`, keeping the higher-confidence text

This eliminates "Contents Contents" duplication artifacts.

### OpenCV NMS

Replaced the O(n^2) Python greedy NMS with `cv2.dnn.NMSBoxes` which runs in C++. Same input format (xywh bboxes + scores), same results, significantly faster for pages with many detections.

### Local+Global Strategy

For documents with both small text and large headers:
- **Local (tiles)**: high-resolution view catches small text
- **Global (full image)**: sees large objects without fragmentation
- **Merge**: keep small objects from tiles, large objects from global, threshold by area ratio

### Multi-threaded Inference

Python preprocessing (resize, padding, crop extraction) can overlap with NPU inference using `ThreadPoolExecutor`. The Hailo `wait_for_async_ready()` gate ensures only one model runs on the NPU at a time, so threads > 3 provide no benefit.

## Model Details

| Model | File | Input | Output | Purpose |
|-------|------|-------|--------|---------|
| Text Detector | `ocr_det.hef` | 544×960×3 (HWC, uint8) | 544×960×1 (heatmap, float32) | DB text region detection |
| Text Recognizer | `ocr.hef` | 48×320×3 (HWC, uint8) | 1×40×97 (char logits, float32) | CTC character recognition |

Character set: 97 characters (digits, uppercase, lowercase, punctuation, space) with blank token at index 0.

## Adding New Features

### New CLI argument

Add to `parse_args()` in `pipeline.py`, wire into `main()`.

### New tiling strategy

1. Add a function following the signature `(image_h, image_w, ...) -> List[Tuple[int,int,int,int]]`
2. Add a CLI flag to select it
3. Wire into the tiling section of `main()` (around line 853)

### New post-processing step

Insert between edge fusion and NMS in `main()` (around line 916-930). The entries list format is:

```python
{
    "tile_index": int,      # which tile this came from
    "crop_index": int,      # which crop within the tile
    "bbox": [x, y, w, h],  # global page coordinates
    "text": str,            # recognized text
    "score": float,         # confidence score
}
```

### New output format

Add alongside `write_structured_outputs()`. The `page_structure` dict contains:

```python
{
    "lines": [
        {
            "line_id": int,
            "indent_level": int,
            "x_anchor": int,
            "text": str,
            "entries": [...]
        }
    ],
    "entries": [...]  # all entries sorted by position
}
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `hailo_platform` | NPU inference (HEF loading, VDevice, async run) |
| `numpy` | Array operations |
| `opencv-python` | Image processing, NMS, visualization |
| `pypdfium2` | PDF rendering |
| `shapely` | Polygon area computation (DB postprocess) |
| `pyclipper` | Vatti polygon clipping (DB postprocess) |
| `symspellpy` | Spelling correction |
| `PyYAML` | Config file parsing |
| `Pillow` | Image format support |
| `degirum-tools` | `generate_tiles_fixed_size()` for aspect-ratio tiling |
