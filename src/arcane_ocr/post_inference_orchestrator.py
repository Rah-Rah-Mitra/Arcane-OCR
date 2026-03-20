from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np


def _as_xywh_array(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4) in [x, y, w, h] format")
    return arr


def apply_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    """Run IoU-based NMS and return kept indices.

    boxes are expected in [x, y, w, h].
    """
    box_arr = _as_xywh_array(boxes)
    score_arr = np.asarray(scores, dtype=np.float32)

    if box_arr.shape[0] != score_arr.shape[0]:
        raise ValueError("boxes and scores must have matching lengths")
    if box_arr.shape[0] == 0:
        return []

    x1 = box_arr[:, 0]
    y1 = box_arr[:, 1]
    x2 = x1 + np.maximum(box_arr[:, 2], 0.0)
    y2 = y1 + np.maximum(box_arr[:, 3], 0.0)
    area = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)

    order = np.argsort(score_arr)[::-1]
    keep: List[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = area[current] + area[rest] - inter
        iou = np.divide(inter, np.maximum(union, 1e-6), dtype=np.float32)
        rest = rest[iou <= float(iou_threshold)]
        order = rest

    return keep


class OCRPostProcessor:
    """Post-inference orchestrator for vertically tiled OCR pages.

    Expected tile layout:
    - Horizontal strips (1D vertical tiling)
    - 25% overlap by default
    """

    def __init__(
        self,
        overlap_ratio: float = 0.25,
        iou_threshold: float = 0.4,
        line_center_threshold_ratio: float = 0.5,
        right_column_ratio: float = 0.72,
    ) -> None:
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0, 1)")
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError("iou_threshold must be in [0, 1]")

        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        self.line_center_threshold_ratio = line_center_threshold_ratio
        self.right_column_ratio = right_column_ratio

    def map_local_to_global(
        self,
        local_boxes: Sequence[Sequence[float]],
        tile_y_offset: int,
    ) -> np.ndarray:
        """Map local tile boxes [x, y, w, h] back to page coordinates."""
        local_arr = _as_xywh_array(local_boxes)
        if local_arr.shape[0] == 0:
            return local_arr
        global_arr = local_arr.copy()
        global_arr[:, 1] += float(tile_y_offset)
        return global_arr

    def _infer_tile_offsets(self, tile_count: int, tile_height: int) -> np.ndarray:
        stride = max(1, int(round(tile_height * (1.0 - self.overlap_ratio))))
        return np.arange(tile_count, dtype=np.int32) * stride

    def collect_global_detections(
        self,
        strip_detections: Sequence[Sequence[Dict[str, Any]]],
        tile_height: int,
    ) -> List[Dict[str, Any]]:
        """Merge local strip detections into global coordinates.

        strip_detections is a list of strips. Each strip is a list of dicts:
        {
            "bbox": [x, y, w, h],
            "score": float,
            "text": "optional"
        }
        """
        offsets = self._infer_tile_offsets(len(strip_detections), tile_height)
        merged: List[Dict[str, Any]] = []

        for strip_idx, strip in enumerate(strip_detections):
            y_offset = int(offsets[strip_idx])
            if not strip:
                continue

            boxes = [item.get("bbox", [0, 0, 0, 0]) for item in strip]
            global_boxes = self.map_local_to_global(boxes, y_offset)

            for local_idx, item in enumerate(strip):
                gx, gy, gw, gh = global_boxes[local_idx].tolist()
                merged.append(
                    {
                        "bbox": [int(round(gx)), int(round(gy)), int(round(gw)), int(round(gh))],
                        "score": float(item.get("score", 0.0)),
                        "text": str(item.get("text", "")).strip(),
                        "strip_index": strip_idx,
                        "strip_y_offset": y_offset,
                    }
                )

        return merged

    def deduplicate(self, detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            return []

        boxes = [det["bbox"] for det in detections]
        scores = [float(det.get("score", 0.0)) for det in detections]
        keep_idx = apply_nms(boxes, scores, self.iou_threshold)

        kept = [detections[i] for i in keep_idx]
        kept.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
        return kept

    @staticmethod
    def _line_bbox(tokens: Sequence[Dict[str, Any]]) -> List[int]:
        xs = [t["bbox"][0] for t in tokens]
        ys = [t["bbox"][1] for t in tokens]
        x2s = [t["bbox"][0] + t["bbox"][2] for t in tokens]
        y2s = [t["bbox"][1] + t["bbox"][3] for t in tokens]
        return [min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys)]

    def group_into_lines(self, detections: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not detections:
            return []

        sorted_det = sorted(detections, key=lambda d: (d["bbox"][1], d["bbox"][0]))
        heights = np.asarray([max(1, int(d["bbox"][3])) for d in sorted_det], dtype=np.float32)
        avg_h = float(np.mean(heights)) if heights.size else 10.0
        center_threshold = max(2.0, avg_h * float(self.line_center_threshold_ratio))

        lines: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_center = 0.0

        for det in sorted_det:
            y = float(det["bbox"][1])
            h = float(det["bbox"][3])
            center = y + 0.5 * h

            if not current:
                current = [det]
                current_center = center
                continue

            if abs(center - current_center) <= center_threshold:
                current.append(det)
                centers = [float(t["bbox"][1]) + 0.5 * float(t["bbox"][3]) for t in current]
                current_center = float(np.mean(np.asarray(centers, dtype=np.float32)))
            else:
                current.sort(key=lambda d: d["bbox"][0])
                lines.append(current)
                current = [det]
                current_center = center

        if current:
            current.sort(key=lambda d: d["bbox"][0])
            lines.append(current)

        return lines

    def _split_title_and_page(
        self,
        line_tokens: Sequence[Dict[str, Any]],
        page_width: int,
    ) -> Dict[str, str]:
        if not line_tokens:
            return {"title": "", "page_number": ""}

        x_boundary = float(page_width) * float(self.right_column_ratio)
        left: List[Dict[str, Any]] = []
        right: List[Dict[str, Any]] = []

        for token in line_tokens:
            x, _, w, _ = token["bbox"]
            center_x = float(x) + 0.5 * float(w)
            if center_x >= x_boundary:
                right.append(token)
            else:
                left.append(token)

        if not right and len(line_tokens) > 1:
            gaps = []
            for i in range(len(line_tokens) - 1):
                left_x, _, left_w, _ = line_tokens[i]["bbox"]
                right_x, _, _, _ = line_tokens[i + 1]["bbox"]
                gap = float(right_x) - float(left_x + left_w)
                gaps.append(gap)
            if gaps:
                max_gap_idx = int(np.argmax(np.asarray(gaps, dtype=np.float32)))
                if gaps[max_gap_idx] >= 0.12 * float(page_width):
                    left = list(line_tokens[: max_gap_idx + 1])
                    right = list(line_tokens[max_gap_idx + 1 :])

        title = " ".join(t.get("text", "").strip() for t in left if t.get("text", "").strip())
        page_raw = " ".join(t.get("text", "").strip() for t in right if t.get("text", "").strip())

        page_match = re.findall(r"[A-Za-z0-9]+", page_raw)
        page_number = "".join(page_match[-2:]) if page_match else ""

        return {"title": title.strip(), "page_number": page_number.strip()}

    def _build_indent_levels(self, lines: Sequence[List[Dict[str, Any]]]) -> List[int]:
        if not lines:
            return []

        anchors = [int(min(tok["bbox"][0] for tok in line)) for line in lines]
        anchors_sorted = sorted(set(anchors))
        if not anchors_sorted:
            return [0] * len(lines)

        if len(anchors_sorted) == 1:
            return [0] * len(lines)

        diffs = np.diff(np.asarray(anchors_sorted, dtype=np.float32))
        spacing = float(np.median(diffs)) if diffs.size else 20.0
        threshold = max(12.0, 0.5 * spacing)

        levels: List[int] = []
        level_anchors: List[int] = []
        for anchor in anchors:
            assigned = -1
            for lvl, lvl_anchor in enumerate(level_anchors):
                if abs(float(anchor) - float(lvl_anchor)) <= threshold:
                    assigned = lvl
                    break
            if assigned == -1:
                level_anchors.append(anchor)
                assigned = len(level_anchors) - 1
            levels.append(assigned)

        return levels

    def build_structure(
        self,
        global_detections: Sequence[Dict[str, Any]],
        page_width: int,
    ) -> Dict[str, Any]:
        lines = self.group_into_lines(global_detections)
        indent_levels = self._build_indent_levels(lines)

        structured_lines: List[Dict[str, Any]] = []
        for line_idx, tokens in enumerate(lines, start=1):
            split = self._split_title_and_page(tokens, page_width)
            line_text = " ".join(t.get("text", "").strip() for t in tokens if t.get("text", "").strip())
            structured_lines.append(
                {
                    "line_id": line_idx,
                    "indent_level": int(indent_levels[line_idx - 1]),
                    "title": split["title"],
                    "page_number": split["page_number"],
                    "text": line_text,
                    "bbox": self._line_bbox(tokens),
                    "tokens": list(tokens),
                }
            )

        markdown = restructure_to_markdown(structured_lines)

        return {
            "type": "contents_page",
            "line_count": len(structured_lines),
            "lines": structured_lines,
            "markdown": markdown,
        }

    def process_tiles(
        self,
        strip_detections: Sequence[Sequence[Dict[str, Any]]],
        tile_height: int,
        page_width: int,
    ) -> Dict[str, Any]:
        global_dets = self.collect_global_detections(strip_detections, tile_height)
        deduped = self.deduplicate(global_dets)
        structured = self.build_structure(deduped, page_width)
        structured["global_detections"] = deduped
        return structured


def restructure_to_markdown(sorted_boxes: Sequence[Dict[str, Any]]) -> str:
    """Create markdown list from structured OCR lines.

    Expected input item format:
    {
      "indent_level": int,
      "title": str,
      "page_number": str,
      "text": str
    }
    """
    lines: List[str] = []

    for item in sorted_boxes:
        level = int(item.get("indent_level", 0))
        title = str(item.get("title", "")).strip()
        page_number = str(item.get("page_number", "")).strip()
        fallback_text = str(item.get("text", "")).strip()

        content = title if title else fallback_text
        if not content:
            continue

        if page_number:
            content = f"{content} {page_number}"

        lines.append(f"{'  ' * max(0, level)}- {content}")

    return "\n".join(lines) + ("\n" if lines else "")
