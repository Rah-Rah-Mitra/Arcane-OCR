#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pypdfium2 as pdfium
import yaml

from .hailo_inference import HailoInfer
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
    parser.add_argument("--pdf-scale", type=float, default=2.0, help="PDF render scale")
    parser.add_argument("--tile-size", type=int, default=1024, help="Sliding-window tile size in pixels")
    parser.add_argument(
        "--tile-overlap-ratio",
        type=float,
        default=0.12,
        help="Tile overlap ratio in [0,1). Example: 0.12 means 12% overlap",
    )
    parser.add_argument("--disable-tiling", action="store_true", help="Disable tiling and run on whole page")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5, help="IoU threshold for duplicate suppression")
    parser.add_argument("--max-pages", type=int, default=0, help="Limit PDF pages (0 means all pages)")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_input_items(input_path: Path, pdf_scale: float, max_pages: int) -> List[Tuple[str, np.ndarray]]:
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
            for page_idx in range(page_total):
                page = pdf[page_idx]
                pil_image = page.render(scale=pdf_scale).to_pil()
                bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                items.append((f"{candidate.stem}_page_{page_idx + 1:02d}", bgr))

    if not items:
        raise RuntimeError(f"No supported image/PDF inputs found under: {input_path}")

    return items


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


def generate_sliding_windows(
    image_h: int,
    image_w: int,
    tile_size: int,
    overlap_ratio: float,
) -> List[Tuple[int, int, int, int]]:
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
    if not entries:
        return []

    sorted_entries = sorted(entries, key=lambda e: float(e.get("score", 0.0)), reverse=True)
    kept: List[Dict] = []

    for candidate in sorted_entries:
        candidate_box = candidate["bbox"]
        suppress = False
        for existing in kept:
            if box_iou_xywh(candidate_box, existing["bbox"]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)

    return sorted(kept, key=lambda e: (e["bbox"][1], e["bbox"][0]))


def infer_tile_entries(
    det_infer: HailoInfer,
    rec_infer: HailoInfer,
    tile_image: np.ndarray,
    tile_origin_x: int,
    tile_origin_y: int,
    det_w: int,
    det_h: int,
    tile_index: int,
) -> List[Dict]:
    det_input = default_preprocess(tile_image, det_w, det_h)
    det_out = run_single_inference(det_infer, det_input)
    crops, local_boxes = det_postprocess(det_out, tile_image, det_h, det_w)

    tile_entries: List[Dict] = []
    for crop_idx, (crop, local_box) in enumerate(zip(crops, local_boxes)):
        rec_input = resize_with_padding(crop)
        rec_out = run_single_inference(rec_infer, rec_input)
        text, conf = ocr_eval_postprocess(rec_out)[0]
        text = text.strip()
        if not text:
            continue

        x, y, w, h = [int(v) for v in local_box]
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

    dictionary_path = Path(config.get("runtime", {}).get("dictionary_path", "./public/samples/frequency_dictionary_en_82_765.txt")).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not det_hef.exists() or not rec_hef.exists():
        print("[ERROR] Missing HEF files.")
        print(f"  Detector: {det_hef}")
        print(f"  Recognizer: {rec_hef}")
        print("Run scripts/download_models.sh or pass --det-hef/--rec-hef.")
        return 1

    input_items = resolve_input_items(Path(args.input).expanduser().resolve(), args.pdf_scale, args.max_pages)

    ocr_corrector = OcrCorrector(str(dictionary_path)) if use_corrector else None

    det_infer = HailoInfer(str(det_hef), batch_size=1)
    rec_infer = HailoInfer(str(rec_hef), batch_size=1, priority=1)

    det_h, det_w, _ = det_infer.get_input_shape()

    print(f"[INFO] Detector HEF: {det_hef}")
    print(f"[INFO] Recognizer HEF: {rec_hef}")
    print(f"[INFO] Input items: {len(input_items)}")

    summary = []
    start_total = time.perf_counter()

    try:
        for item_name, original in input_items:
            start_item = time.perf_counter()

            page_h, page_w = original.shape[:2]
            use_tiling = not args.disable_tiling

            if use_tiling:
                windows = generate_sliding_windows(
                    image_h=page_h,
                    image_w=page_w,
                    tile_size=max(128, int(args.tile_size)),
                    overlap_ratio=float(args.tile_overlap_ratio),
                )
            else:
                windows = [(0, 0, page_w, page_h)]

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
                )
                all_entries.extend(tile_entries)

            dedup_entries = nms_entries(all_entries, iou_threshold=float(args.nms_iou_threshold))
            boxes = [entry["bbox"] for entry in dedup_entries]
            texts = [entry["text"] for entry in dedup_entries]

            annotated = visualize_ocr_annotations(original, boxes, texts, ocr_corrector)
            out_path = output_dir / f"{item_name}_ocr.png"
            cv2.imwrite(str(out_path), annotated)

            page_structure = build_page_structure(dedup_entries, page_w=page_w, page_h=page_h)
            structured_json, structured_md = write_structured_outputs(item_name, page_structure, output_dir)

            elapsed = time.perf_counter() - start_item
            print(
                f"[TIME] {item_name}: {elapsed:.3f}s "
                f"({len(boxes)} text regions, {len(windows)} tiles, overlap={args.tile_overlap_ratio:.2f})"
            )

            summary.append(
                {
                    "item": item_name,
                    "elapsed_seconds": round(elapsed, 4),
                    "text_regions": len(boxes),
                    "tiles": len(windows),
                    "tile_size": int(args.tile_size),
                    "tile_overlap_ratio": float(args.tile_overlap_ratio),
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
