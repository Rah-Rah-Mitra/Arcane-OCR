#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium
import yaml
from degirum_tools import generate_tiles_fixed_size

from .hailo_inference import HailoInfer, create_shared_vdevice
from .ocr_utils import (
    OcrCorrector,
    default_preprocess,
    det_postprocess,
    ocr_eval_postprocess,
    resize_with_padding,
    visualize_ocr_annotations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone Hailo OCR for image and PDF inputs")
    parser.add_argument("--input", "-i", required=True, help="Input image, PDF, or directory")
    parser.add_argument("--config", default="./config/ocr_config.yaml", help="Path to OCR config YAML")
    parser.add_argument("--det-hef", default=None, help="Override detector HEF path")
    parser.add_argument("--rec-hef", default=None, help="Override recognizer HEF path")
    parser.add_argument("--output-dir", default="./output", help="Directory to save outputs")
    parser.add_argument("--use-corrector", action="store_true", help="Enable OCR text correction")
    parser.add_argument("--disable-corrector", action="store_true", help="Disable OCR text correction")
    parser.add_argument("--dictionary-path", default=None, help="Path to custom SymSpell frequency dictionary (overrides config)")
    parser.add_argument("--pdf-scale", type=float, default=2.0, help="PDF render scale")
    parser.add_argument("--tile-size", type=int, default=1024, help="Sliding-window tile size in pixels")
    parser.add_argument(
        "--tile-overlap-ratio",
        type=float,
        default=0.12,
        help="Tile overlap ratio in [0,1). Example: 0.12 means 12%% overlap",
    )
    parser.add_argument("--disable-tiling", action="store_true", help="Disable tiling and run on whole page")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="IoU threshold for duplicate suppression")
    parser.add_argument(
        "--max-tiles-per-page",
        type=int,
        default=9,
        help="Adaptive tiling budget (0 disables adaptation and uses fixed tile-size)",
    )
    parser.add_argument("--a4-mode", action="store_true", help="A4-optimized: render at 200 DPI, tile to detector native 544x960")
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=200,
        help="PDF render DPI (used if --a4-mode or no --pdf-scale). Standard: 150 (fast), 200 (balanced), 300 (high-res)",
    )
    parser.add_argument("--det-tile-height", type=int, default=544, help="Detector tile height (native: 544)")
    parser.add_argument("--det-tile-width", type=int, default=960, help="Detector tile width (native: 960)")
    parser.add_argument("--rec-batch-size", type=int, default=8, help="Recognizer batch size per inference call")
    parser.add_argument("--min-box-area", type=int, default=24, help="Skip detector boxes smaller than this area")
    parser.add_argument("--det-priority", type=int, default=0, help="Scheduler priority for detector model")
    parser.add_argument("--rec-priority", type=int, default=1, help="Scheduler priority for recognizer model")
    parser.add_argument("--group-id", default="SHARED", help="Hailo vdevice group id")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit PDF pages (0 means all pages)")
    # --- Advanced tiling strategies ---
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=0.03,
        help="Edge band as fraction of tile dimension for box fusion (0.03 = 3%%)",
    )
    parser.add_argument(
        "--fusion-threshold",
        type=float,
        default=0.5,
        help="1D IoU threshold for fusing split edge detections",
    )
    parser.add_argument("--disable-edge-fusion", action="store_true", help="Disable edge box fusion")
    parser.add_argument("--enable-local-global", action="store_true", help="Enable local+global tiling strategy")
    parser.add_argument(
        "--large-object-threshold",
        type=float,
        default=0.005,
        help="Area ratio threshold for local vs global object assignment",
    )
    parser.add_argument(
        "--tile-workers",
        type=int,
        default=1,
        help="Number of parallel tile inference threads (1=sequential, 2-3 for overlap)",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dpi_to_scale(dpi: float, base_dpi: float = 72.0) -> float:
    return dpi / base_dpi


def resolve_input_items(
    input_path: Path,
    pdf_scale: float,
    max_pages: int,
    use_a4_mode: bool = False,
    pdf_dpi: int = 200,
) -> List[Tuple[str, np.ndarray]]:
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}

    def _is_pdf(path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    items: List[Tuple[str, np.ndarray]] = []

    if input_path.is_file():
        candidates = [input_path]
    elif input_path.is_dir():
        candidates = sorted([p for p in input_path.rglob("*") if p.is_file() and (_is_image(p) or _is_pdf(p))])
    else:
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    for candidate in candidates:
        if _is_image(candidate):
            image = cv2.imread(str(candidate))
            if image is None:
                print(f"[WARN] Skipping unreadable image: {candidate}")
                continue
            items.append((candidate.stem, image))
        elif _is_pdf(candidate):
            pdf = pdfium.PdfDocument(str(candidate))
            page_total = len(pdf)
            if max_pages > 0:
                page_total = min(page_total, max_pages)
            effective_scale = dpi_to_scale(pdf_dpi) if use_a4_mode else pdf_scale
            for page_idx in range(page_total):
                page = pdf[page_idx]
                pil_image = page.render(scale=effective_scale).to_pil()
                bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                items.append((f"{candidate.stem}_page_{page_idx + 1:02d}", bgr))

    if not items:
        raise RuntimeError(f"No supported image/PDF inputs found under: {input_path}")

    return items


# ---------------------------------------------------------------------------
# Inference helpers (unchanged)
# ---------------------------------------------------------------------------


def run_single_inference(hailo_infer: HailoInfer, image: np.ndarray) -> np.ndarray:
    holder: Dict[str, np.ndarray | Exception] = {}

    def callback(completion_info, bindings_list, **kwargs):
        if completion_info.exception:
            holder["error"] = completion_info.exception
            return
        holder["result"] = bindings_list[0].output().get_buffer()

    job = hailo_infer.run([image], callback)
    job.wait(10000)

    if "error" in holder:
        raise RuntimeError(f"Inference failed: {holder['error']}")
    if "result" not in holder:
        raise RuntimeError("Inference callback did not return output")

    return holder["result"]  # type: ignore[return-value]


def run_batch_inference(hailo_infer: HailoInfer, images: List[np.ndarray]) -> List[np.ndarray]:
    if not images:
        return []

    holder: Dict[str, List[np.ndarray] | Exception] = {}

    def callback(completion_info, bindings_list, **kwargs):
        if completion_info.exception:
            holder["error"] = completion_info.exception
            return
        holder["result"] = [binding.output().get_buffer() for binding in bindings_list]

    job = hailo_infer.run(images, callback)
    job.wait(10000)

    if "error" in holder:
        raise RuntimeError(f"Inference failed: {holder['error']}")
    if "result" not in holder:
        raise RuntimeError("Inference callback did not return output")

    return holder["result"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tiling strategies
# ---------------------------------------------------------------------------


def generate_sliding_windows(
    image_h: int,
    image_w: int,
    tile_size: int,
    overlap_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    """Legacy square-tile sliding window (kept as fallback)."""
    overlap_ratio = max(0.0, min(overlap_ratio, 0.95))
    stride = max(1, int(tile_size * (1.0 - overlap_ratio)))

    if image_h <= tile_size:
        ys = [0]
    else:
        ys = list(range(0, image_h - tile_size + 1, stride))
        if ys[-1] != image_h - tile_size:
            ys.append(image_h - tile_size)

    if image_w <= tile_size:
        xs = [0]
    else:
        xs = list(range(0, image_w - tile_size + 1, stride))
        if xs[-1] != image_w - tile_size:
            xs.append(image_w - tile_size)

    windows: List[Tuple[int, int, int, int]] = []
    for y in ys:
        for x in xs:
            x2 = min(image_w, x + tile_size)
            y2 = min(image_h, y + tile_size)
            windows.append((x, y, x2, y2))
    return windows


def generate_aspect_ratio_windows(
    image_h: int,
    image_w: int,
    tile_w: int,
    tile_h: int,
    overlap_percent: float,
) -> List[Tuple[int, int, int, int]]:
    """Generate tiling windows matching the detector's native aspect ratio.

    Uses degirum_tools.generate_tiles_fixed_size() for proper 2D grid
    generation with independent width/height and evenly-distributed overlap.
    """
    if image_w <= tile_w and image_h <= tile_h:
        return [(0, 0, image_w, image_h)]

    eff_tile_w = min(tile_w, image_w)
    eff_tile_h = min(tile_h, image_h)

    tiles_grid = generate_tiles_fixed_size(
        tile_size=(eff_tile_w, eff_tile_h),
        image_size=(image_w, image_h),
        min_overlap_percent=(overlap_percent, overlap_percent),
    )
    windows: List[Tuple[int, int, int, int]] = []
    for row_idx in range(tiles_grid.shape[0]):
        for col_idx in range(tiles_grid.shape[1]):
            x1, y1, x2, y2 = tiles_grid[row_idx, col_idx]
            windows.append((int(x1), int(y1), int(x2), int(y2)))
    return windows


def adapt_tile_size_for_budget(
    image_h: int,
    image_w: int,
    base_tile_size: int,
    overlap_ratio: float,
    max_tiles_per_page: int,
) -> int:
    tile_size = max(128, int(base_tile_size))
    if max_tiles_per_page <= 0:
        return tile_size

    max_dim = max(image_h, image_w)
    if image_h <= tile_size and image_w <= tile_size:
        return tile_size

    while tile_size <= max_dim:
        windows = generate_sliding_windows(image_h, image_w, tile_size, overlap_ratio)
        if len(windows) <= max_tiles_per_page:
            return tile_size
        tile_size += 128

    return max_dim


# ---------------------------------------------------------------------------
# NMS (OpenCV-accelerated)
# ---------------------------------------------------------------------------


def box_iou_xywh(box_a: List[int], box_b: List[int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    union = area_a + area_b - inter_area
    return float(inter_area / max(1, union))


def nms_entries(entries: List[Dict], iou_threshold: float) -> List[Dict]:
    """Suppress duplicate detections using OpenCV NMS.

    Uses cv2.dnn.NMSBoxes for fast C++ NMS instead of O(n^2) Python loop.
    Bboxes are in [x, y, w, h] format which cv2.dnn.NMSBoxes expects.
    """
    if not entries:
        return []

    bboxes = [entry["bbox"] for entry in entries]
    scores = [float(entry.get("score", 0.0)) for entry in entries]

    indices = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(indices) == 0:
        return []

    indices = np.array(indices).flatten()
    kept = [entries[i] for i in indices]
    return sorted(kept, key=lambda e: (e["bbox"][1], e["bbox"][0]))


# ---------------------------------------------------------------------------
# Edge box fusion (from 017_advanced_tiling_strategies notebook)
# ---------------------------------------------------------------------------


def _compute_grid_dims(windows: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
    """Infer (n_rows, n_cols) from a row-major list of tile windows."""
    if not windows:
        return (1, 1)
    first_y = windows[0][1]
    n_cols = 0
    for w in windows:
        if w[1] == first_y:
            n_cols += 1
        else:
            break
    n_cols = max(1, n_cols)
    n_rows = max(1, len(windows) // n_cols)
    return (n_rows, n_cols)


def _max_1d_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute max(iou_x, iou_y) for two boxes in [x1,y1,x2,y2] format.

    Two text fragments split by a tile edge will have high 1D IoU in the
    axis parallel to the split (e.g. high iou_y for a vertical tile edge
    cutting a horizontal text line).
    """
    inter_x = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    inter_y = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))

    if min(inter_x, inter_y) == 0:
        return 0.0

    wa = box_a[2] - box_a[0]
    wb = box_b[2] - box_b[0]
    ha = box_a[3] - box_a[1]
    hb = box_b[3] - box_b[1]

    iou_x = inter_x / max(wa + wb - inter_x, 1e-6)
    iou_y = inter_y / max(ha + hb - inter_y, 1e-6)

    return max(iou_x, iou_y)


def classify_edge_entries(
    entries: List[Dict],
    tile_windows: List[Tuple[int, int, int, int]],
    edge_threshold: float = 0.03,
    n_cols: int = 1,
    n_rows: int = 1,
) -> Tuple[List[Dict], List[Dict]]:
    """Separate entries into central and edge detections.

    Edge entries are those whose bounding box falls within the edge band
    of an interior tile edge (edges that border another tile, not image
    boundaries).
    """
    central: List[Dict] = []
    edge: List[Dict] = []

    for entry in entries:
        tile_idx = entry.get("tile_index", 0)
        if tile_idx < 0 or tile_idx >= len(tile_windows):
            central.append(entry)
            continue

        tx1, ty1, tx2, ty2 = tile_windows[tile_idx]
        tile_w = tx2 - tx1
        tile_h = ty2 - ty1

        bx, by, bw, bh = entry["bbox"]
        gx1, gy1, gx2, gy2 = bx, by, bx + bw, by + bh

        row = tile_idx // n_cols
        col = tile_idx % n_cols
        has_top = row > 0
        has_bot = row < n_rows - 1
        has_left = col > 0
        has_right = col < n_cols - 1

        margin_x = int(edge_threshold * tile_w)
        margin_y = int(edge_threshold * tile_h)

        is_edge = False
        if has_top and gy1 < ty1 + margin_y:
            is_edge = True
        if has_bot and gy2 > ty2 - margin_y:
            is_edge = True
        if has_left and gx1 < tx1 + margin_x:
            is_edge = True
        if has_right and gx2 > tx2 - margin_x:
            is_edge = True

        if is_edge:
            e = dict(entry)
            e["_xyxy"] = [float(gx1), float(gy1), float(gx2), float(gy2)]
            edge.append(e)
        else:
            central.append(entry)

    return central, edge


def fuse_edge_entries(
    edge_entries: List[Dict],
    fusion_threshold: float = 0.5,
) -> List[Dict]:
    """Fuse split text detections at tile edges using 1D IoU.

    Implements the edge_box_fusion algorithm: for each pair of edge
    detections, computes max(iou_x, iou_y) and merges if above threshold.
    Picks the higher-confidence text for the merged result.
    """
    if not edge_entries:
        return []

    sorted_entries = sorted(edge_entries, key=lambda e: -e["score"])

    # First pass: group by 1D IoU (max 2 per group)
    groups: List[List[Dict]] = []
    used = [False] * len(sorted_entries)

    for i, entry_a in enumerate(sorted_entries):
        if used[i]:
            continue
        group = [entry_a]
        used[i] = True
        for j in range(i + 1, len(sorted_entries)):
            if used[j]:
                continue
            if len(group) >= 2:
                break
            iou_1d = _max_1d_iou(entry_a["_xyxy"], sorted_entries[j]["_xyxy"])
            if iou_1d > fusion_threshold:
                group.append(sorted_entries[j])
                used[j] = True
        groups.append(group)

    # Merge each group
    fused: List[Dict] = []
    for group in groups:
        if len(group) == 1:
            e = group[0]
            xyxy = e["_xyxy"]
            fused.append({
                "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])],
                "text": e["text"],
                "score": e["score"],
                "tile_index": e["tile_index"],
                "crop_index": e.get("crop_index", -1),
            })
        else:
            a, b = group[0], group[1]
            ax, bx = a["_xyxy"], b["_xyxy"]
            mx1 = min(ax[0], bx[0])
            my1 = min(ax[1], bx[1])
            mx2 = max(ax[2], bx[2])
            my2 = max(ax[3], bx[3])
            text = a["text"] if a["score"] >= b["score"] else b["text"]
            score = (a["score"] + b["score"]) / 2.0
            fused.append({
                "bbox": [int(mx1), int(my1), int(mx2 - mx1), int(my2 - my1)],
                "text": text,
                "score": score,
                "tile_index": a["tile_index"],
                "crop_index": a.get("crop_index", -1),
            })

    return fused


# ---------------------------------------------------------------------------
# Per-tile inference (unchanged)
# ---------------------------------------------------------------------------


def infer_tile_entries(
    det_infer: HailoInfer,
    rec_infer: HailoInfer,
    tile_image: np.ndarray,
    tile_origin_x: int,
    tile_origin_y: int,
    det_w: int,
    det_h: int,
    tile_index: int,
    rec_batch_size: int,
    min_box_area: int,
) -> List[Dict]:
    det_input = default_preprocess(tile_image, det_w, det_h)
    det_out = run_single_inference(det_infer, det_input)
    crops, local_boxes = det_postprocess(det_out, tile_image, det_h, det_w)

    tile_entries: List[Dict] = []
    rec_inputs: List[np.ndarray] = []
    valid_meta: List[Tuple[int, List[int]]] = []

    for crop_idx, (crop, local_box) in enumerate(zip(crops, local_boxes)):
        x, y, w, h = [int(v) for v in local_box]
        if w <= 0 or h <= 0 or (w * h) < min_box_area:
            continue
        rec_inputs.append(resize_with_padding(crop))
        valid_meta.append((crop_idx, [x, y, w, h]))

    batch_size = max(1, int(rec_batch_size))
    for start in range(0, len(rec_inputs), batch_size):
        batch_inputs = rec_inputs[start:start + batch_size]
        batch_meta = valid_meta[start:start + batch_size]
        actual_count = len(batch_inputs)

        if actual_count < batch_size:
            pad_input = batch_inputs[-1]
            while len(batch_inputs) < batch_size:
                batch_inputs.append(pad_input)
                batch_meta.append((-1, [0, 0, 0, 0]))

        batch_out = run_batch_inference(rec_infer, batch_inputs)

        for rec_out, (crop_idx, local_box) in zip(batch_out[:actual_count], batch_meta[:actual_count]):
            text, conf = ocr_eval_postprocess(rec_out)[0]
            text = text.strip()
            if not text:
                continue

            x, y, w, h = local_box
            global_bbox = [tile_origin_x + x, tile_origin_y + y, w, h]
            tile_entries.append(
                {
                    "tile_index": tile_index,
                    "crop_index": crop_idx,
                    "bbox": global_bbox,
                    "text": text,
                    "score": float(conf),
                }
            )

    return tile_entries


# ---------------------------------------------------------------------------
# Local+Global strategy (from 017_advanced_tiling_strategies notebook)
# ---------------------------------------------------------------------------


def run_local_global_inference(
    det_infer: HailoInfer,
    rec_infer: HailoInfer,
    original: np.ndarray,
    windows: List[Tuple[int, int, int, int]],
    det_w: int,
    det_h: int,
    rec_batch_size: int,
    min_box_area: int,
    large_object_threshold: float = 0.005,
) -> List[Dict]:
    """Run local (tile) + global (full-image) inference and merge by object size.

    Large detections (area >= threshold * page_area) from tiles are discarded
    in favor of the global inference, which sees them without fragmentation.
    Small detections from the global inference are discarded in favor of tile
    results, which have higher resolution.
    """
    page_h, page_w = original.shape[:2]
    page_area = page_w * page_h

    local_entries: List[Dict] = []
    for tile_idx, (x1, y1, x2, y2) in enumerate(windows):
        tile = original[y1:y2, x1:x2]
        tile_entries = infer_tile_entries(
            det_infer, rec_infer, tile, x1, y1,
            det_w, det_h, tile_idx, rec_batch_size, min_box_area,
        )
        local_entries.extend(tile_entries)

    area_threshold = large_object_threshold * page_area
    local_filtered = [
        e for e in local_entries
        if e["bbox"][2] * e["bbox"][3] < area_threshold
    ]

    global_entries = infer_tile_entries(
        det_infer, rec_infer, original, 0, 0,
        det_w, det_h, tile_index=-1,
        rec_batch_size=rec_batch_size, min_box_area=min_box_area,
    )

    global_min_threshold = large_object_threshold * 0.5
    global_filtered = [
        e for e in global_entries
        if e["bbox"][2] * e["bbox"][3] >= global_min_threshold * page_area
    ]

    return local_filtered + global_filtered


# ---------------------------------------------------------------------------
# Multi-threaded tile inference (from 006_multi_threading notebook)
# ---------------------------------------------------------------------------


def infer_tiles_parallel(
    det_infer: HailoInfer,
    rec_infer: HailoInfer,
    original: np.ndarray,
    windows: List[Tuple[int, int, int, int]],
    det_w: int,
    det_h: int,
    rec_batch_size: int,
    min_box_area: int,
    max_workers: int = 2,
) -> List[Dict]:
    """Process tiles in parallel using a thread pool.

    Overlaps Python preprocessing with Hailo inference. Keep max_workers
    at 2-3 since the HailoInfer's wait_for_async_ready gate serializes
    actual accelerator access.
    """
    all_entries: List[Dict] = []

    def _process_tile(tile_idx: int, x1: int, y1: int, x2: int, y2: int) -> List[Dict]:
        tile = original[y1:y2, x1:x2]
        return infer_tile_entries(
            det_infer, rec_infer, tile, x1, y1,
            det_w, det_h, tile_idx, rec_batch_size, min_box_area,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_tile, idx, x1, y1, x2, y2): idx
            for idx, (x1, y1, x2, y2) in enumerate(windows)
        }
        for future in as_completed(futures):
            tile_entries = future.result()
            all_entries.extend(tile_entries)

    return all_entries


# ---------------------------------------------------------------------------
# Structured output and visualization (unchanged)
# ---------------------------------------------------------------------------


def build_page_structure(entries: List[Dict], page_w: int, page_h: int) -> Dict:
    if not entries:
        return {"lines": [], "entries": []}

    sorted_entries = sorted(entries, key=lambda e: (e["bbox"][1], e["bbox"][0]))
    line_tolerance = max(12, int(0.015 * page_h))

    lines: List[Dict] = []
    current: List[Dict] = []
    current_center_y: float | None = None

    for entry in sorted_entries:
        x, y, w, h = entry["bbox"]
        center_y = y + h / 2.0

        if current_center_y is None:
            current = [entry]
            current_center_y = center_y
            continue

        if abs(center_y - current_center_y) <= line_tolerance:
            current.append(entry)
            current_center_y = float(sum(e["bbox"][1] + e["bbox"][3] / 2.0 for e in current) / len(current))
        else:
            lines.append({"entries": sorted(current, key=lambda e: e["bbox"][0])})
            current = [entry]
            current_center_y = center_y

    if current:
        lines.append({"entries": sorted(current, key=lambda e: e["bbox"][0])})

    left_positions = sorted(set(e["bbox"][0] for e in sorted_entries))
    indent_threshold = max(18, int(0.02 * page_w))
    indent_levels: List[int] = []
    for pos in left_positions:
        if not indent_levels:
            indent_levels.append(pos)
            continue
        if pos - indent_levels[-1] > indent_threshold:
            indent_levels.append(pos)

    def _indent_level(x: int) -> int:
        if not indent_levels:
            return 0
        return min(range(len(indent_levels)), key=lambda i: abs(indent_levels[i] - x))

    structured_lines: List[Dict] = []
    for line_idx, line in enumerate(lines, start=1):
        merged_text = " ".join(e["text"].strip() for e in line["entries"] if e["text"].strip())
        line_x = min(e["bbox"][0] for e in line["entries"])
        level = _indent_level(line_x)
        structured_lines.append(
            {
                "line_id": line_idx,
                "indent_level": int(level),
                "x_anchor": int(line_x),
                "text": merged_text,
                "entries": line["entries"],
            }
        )

    return {"lines": structured_lines, "entries": sorted_entries}


def write_structured_outputs(page_name: str, page_structure: Dict, output_dir: Path) -> Tuple[Path, Path]:
    json_path = output_dir / f"{page_name}_structured.json"
    md_path = output_dir / f"{page_name}_structured.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(page_structure, f, indent=2, ensure_ascii=False)

    md_lines = [f"# OCR Structure: {page_name}", ""]
    for line in page_structure.get("lines", []):
        indent = "  " * int(line.get("indent_level", 0))
        text = line.get("text", "").strip()
        if text:
            md_lines.append(f"{indent}- {text}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config))

    det_hef = Path(args.det_hef or config["models"]["det_hef"]).expanduser().resolve()
    rec_hef = Path(args.rec_hef or config["models"]["rec_hef"]).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    use_corrector = config.get("runtime", {}).get("use_corrector", True)
    if args.use_corrector:
        use_corrector = True
    if args.disable_corrector:
        use_corrector = False

    dictionary_path = Path(
        args.dictionary_path
        or config.get("runtime", {}).get("dictionary_path", "./public/samples/frequency_dictionary_en_82_765.txt")
    ).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not det_hef.exists() or not rec_hef.exists():
        print("[ERROR] Missing HEF files.")
        print(f"  Detector: {det_hef}")
        print(f"  Recognizer: {rec_hef}")
        print("Run scripts/download_models.sh or pass --det-hef/--rec-hef.")
        return 1

    input_items = resolve_input_items(
        Path(args.input).expanduser().resolve(),
        args.pdf_scale,
        args.max_pages,
        use_a4_mode=args.a4_mode,
        pdf_dpi=int(args.pdf_dpi),
    )

    ocr_corrector = OcrCorrector(str(dictionary_path)) if use_corrector else None

    shared_vdevice = create_shared_vdevice(group_id=args.group_id)
    det_infer = HailoInfer(
        str(det_hef),
        batch_size=1,
        priority=int(args.det_priority),
        vdevice=shared_vdevice,
        group_id=args.group_id,
    )
    rec_infer = HailoInfer(
        str(rec_hef),
        batch_size=max(1, int(args.rec_batch_size)),
        priority=int(args.rec_priority),
        vdevice=shared_vdevice,
        group_id=args.group_id,
    )

    det_h, det_w, _ = det_infer.get_input_shape()

    print(f"[INFO] Detector HEF: {det_hef}")
    print(f"[INFO] Recognizer HEF: {rec_hef}")
    print(f"[INFO] Input items: {len(input_items)}")
    if args.a4_mode:
        print(f"[INFO] A4-mode enabled: PDF rendering at {args.pdf_dpi} DPI, tiling to detector native {det_h}×{det_w}")

    use_edge_fusion = not args.disable_edge_fusion
    use_local_global = args.enable_local_global
    tile_workers = max(1, int(args.tile_workers))

    summary = []
    start_total = time.perf_counter()

    try:
        for item_name, original in input_items:
            start_item = time.perf_counter()

            page_h, page_w = original.shape[:2]
            use_tiling = not args.disable_tiling
            render_mode = "a4-native" if args.a4_mode else "manual-scale"

            if use_tiling:
                if args.a4_mode:
                    effective_tile_h = int(args.det_tile_height)
                    effective_tile_w = int(args.det_tile_width)
                else:
                    effective_tile_w = max(128, int(args.tile_size))
                    effective_tile_h = int(effective_tile_w * det_h / det_w)
                    effective_tile_w = adapt_tile_size_for_budget(
                        image_h=page_h,
                        image_w=page_w,
                        base_tile_size=effective_tile_w,
                        overlap_ratio=float(args.tile_overlap_ratio),
                        max_tiles_per_page=int(args.max_tiles_per_page),
                    )
                    effective_tile_h = int(effective_tile_w * det_h / det_w)

                overlap_pct = float(args.tile_overlap_ratio) * 100.0
                windows = generate_aspect_ratio_windows(
                    image_h=page_h,
                    image_w=page_w,
                    tile_w=effective_tile_w,
                    tile_h=effective_tile_h,
                    overlap_percent=overlap_pct,
                )
            else:
                effective_tile_w = page_w
                effective_tile_h = page_h
                windows = [(0, 0, page_w, page_h)]

            # --- Tile inference ---
            if use_local_global and use_tiling:
                all_entries = run_local_global_inference(
                    det_infer, rec_infer, original, windows,
                    det_w, det_h,
                    int(args.rec_batch_size), max(1, int(args.min_box_area)),
                    large_object_threshold=float(args.large_object_threshold),
                )
            elif tile_workers > 1 and use_tiling:
                all_entries = infer_tiles_parallel(
                    det_infer, rec_infer, original, windows,
                    det_w, det_h,
                    int(args.rec_batch_size), max(1, int(args.min_box_area)),
                    max_workers=tile_workers,
                )
            else:
                all_entries: List[Dict] = []
                for tile_idx, (x1, y1, x2, y2) in enumerate(windows):
                    tile = original[y1:y2, x1:x2]
                    tile_entries = infer_tile_entries(
                        det_infer=det_infer,
                        rec_infer=rec_infer,
                        tile_image=tile,
                        tile_origin_x=x1,
                        tile_origin_y=y1,
                        det_w=det_w,
                        det_h=det_h,
                        tile_index=tile_idx,
                        rec_batch_size=int(args.rec_batch_size),
                        min_box_area=max(1, int(args.min_box_area)),
                    )
                    all_entries.extend(tile_entries)

            # --- Edge box fusion ---
            if use_edge_fusion and use_tiling and len(windows) > 1:
                n_rows, n_cols = _compute_grid_dims(windows)
                central_entries, edge_entries = classify_edge_entries(
                    all_entries, windows,
                    edge_threshold=float(args.edge_threshold),
                    n_cols=n_cols, n_rows=n_rows,
                )
                fused_edge = fuse_edge_entries(
                    edge_entries,
                    fusion_threshold=float(args.fusion_threshold),
                )
                all_entries = central_entries + fused_edge

            # --- NMS ---
            dedup_entries = nms_entries(all_entries, iou_threshold=float(args.nms_iou_threshold))

            # --- SymSpell correction ---
            if ocr_corrector:
                for entry in dedup_entries:
                    entry["text_raw"] = entry["text"]
                    entry["text"] = ocr_corrector.correct_text(entry["text"])

            boxes = [entry["bbox"] for entry in dedup_entries]
            texts = [entry["text"] for entry in dedup_entries]

            annotated = visualize_ocr_annotations(original, boxes, texts, None)
            out_path = output_dir / f"{item_name}_ocr.png"
            cv2.imwrite(str(out_path), annotated)

            page_structure = build_page_structure(dedup_entries, page_w=page_w, page_h=page_h)
            structured_json, structured_md = write_structured_outputs(item_name, page_structure, output_dir)

            elapsed = time.perf_counter() - start_item
            print(
                f"[TIME] {item_name}: {elapsed:.3f}s "
                f"({len(boxes)} regions, {len(windows)} tiles, tile={effective_tile_w}×{effective_tile_h}, "
                f"mode={render_mode}, overlap={args.tile_overlap_ratio:.2f})"
            )

            summary.append(
                {
                    "item": item_name,
                    "elapsed_seconds": round(elapsed, 4),
                    "text_regions": len(boxes),
                    "tiles": len(windows),
                    "tile_width": int(effective_tile_w),
                    "tile_height": int(effective_tile_h),
                    "render_mode": render_mode,
                    "tile_overlap_ratio": float(args.tile_overlap_ratio),
                    "rec_batch_size": int(args.rec_batch_size),
                    "min_box_area": int(args.min_box_area),
                    "output": str(out_path),
                    "structured_json": str(structured_json),
                    "structured_markdown": str(structured_md),
                }
            )

    finally:
        rec_infer.close()
        det_infer.close()

    total_elapsed = time.perf_counter() - start_total
    report_path = output_dir / "timing_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "total_seconds": round(total_elapsed, 4),
                "items_processed": len(summary),
                "details": summary,
            },
            f,
            indent=2,
        )

    print(f"[TIME] Total OCR runtime: {total_elapsed:.3f}s")
    print(f"[INFO] Timing report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
