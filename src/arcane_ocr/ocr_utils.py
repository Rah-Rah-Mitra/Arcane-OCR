from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from symspellpy import SymSpell

from .db_postprocess import DBPostProcess


CHARACTERS = [
    "blank", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
    "{", "|", "}", "~", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " ", " "
]


def ocr_eval_postprocess(
    infer_results: np.ndarray | list,
    character_dict: List[str] | None = None,
) -> List[Tuple[str, float]]:
    return ocr_eval_postprocess_with_dict(infer_results, character_dict=character_dict)


def load_ocr_character_dictionary(dict_path: str) -> List[str]:
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Missing recognition dictionary file: {dict_path}")

    if dict_path.lower().endswith(".npz"):
        payload = np.load(dict_path)
        if "dictionary" not in payload:
            raise ValueError(f"NPZ dictionary file missing 'dictionary' key: {dict_path}")

        chars: List[str] = []
        for token in payload["dictionary"]:
            if isinstance(token, bytes):
                chars.append(token.decode("utf-8", errors="ignore"))
            else:
                chars.append(str(token))
        chars = [c for c in chars if c]
        if not chars:
            raise ValueError(f"Recognition dictionary is empty: {dict_path}")
        return chars

    chars: List[str] = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.rstrip("\r\n")
            if token:
                chars.append(token)

    if not chars:
        raise ValueError(f"Recognition dictionary is empty: {dict_path}")
    return chars


def ocr_eval_postprocess_with_dict(
    infer_results: np.ndarray | list,
    character_dict: List[str] | None,
) -> List[Tuple[str, float]]:
    dict_character = list(character_dict) if character_dict else list(CHARACTERS)
    # Compiled PP-OCRv5 recognizer HEFs expose one blank/control token offset.
    ignored_tokens = [0]
    token_shift = 1 if character_dict else 0
    vocab_size = len(dict_character)

    if isinstance(infer_results, list):
        infer_results = np.array(infer_results)

    if infer_results.ndim == 2:
        infer_results = np.expand_dims(infer_results, axis=0)

    text_prob = infer_results.max(axis=2)
    text_index = infer_results.argmax(axis=2)

    results: List[Tuple[str, float]] = []
    for batch_idx in range(len(text_index)):
        selection = np.ones(len(text_index[batch_idx]), dtype=bool)
        selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]

        for ignored_token in ignored_tokens:
            selection &= text_index[batch_idx] != ignored_token

        char_list = []
        for text_id in text_index[batch_idx][selection]:
            idx = int(text_id)
            mapped_idx = idx - token_shift
            if 0 <= mapped_idx < vocab_size:
                char_list.append(dict_character[mapped_idx])
        conf_list = text_prob[batch_idx][selection] if text_prob is not None else [1] * len(selection)
        if len(conf_list) == 0:
            conf_list = [0]

        text = "".join(char_list)
        results.append((text, float(np.mean(conf_list).tolist())))

    return results


def resize_heatmap_to_original(
    heatmap: np.ndarray,
    original_size: Tuple[int, int],
    model_w: int,
    model_h: int,
) -> np.ndarray:
    orig_h, orig_w = original_size
    scale = min(model_w / orig_w, model_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2

    cropped_heatmap = heatmap[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
    resized_heatmap = cv2.resize(cropped_heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return resized_heatmap


def warp_to_rectangle(image: np.ndarray, poly: np.ndarray) -> np.ndarray:
    poly = poly.astype(np.float32)
    w = int(np.linalg.norm(poly[0] - poly[1]))
    h = int(np.linalg.norm(poly[0] - poly[3]))
    w = max(w, 1)
    h = max(h, 1)
    dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(poly, dst_pts)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)


def get_cropped_text_images(
    heatmap: np.ndarray,
    orig_img: np.ndarray,
    model_height: int,
    model_width: int,
    bin_thresh: float = 0.3,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    postprocess = DBPostProcess(
        thresh=bin_thresh,
        box_thresh=0.6,
        max_candidates=1000,
        unclip_ratio=1.5,
    )

    heatmap_resized = resize_heatmap_to_original(
        heatmap,
        original_size=orig_img.shape[:2],
        model_w=model_width,
        model_h=model_height,
    )

    preds = {"maps": heatmap_resized[None, None, :, :]}
    boxes_batch = postprocess(preds, [(*orig_img.shape[:2], 1.0, 1.0)])
    boxes = boxes_batch[0]["points"]

    cropped_images: List[np.ndarray] = []
    boxes_location: List[List[int]] = []

    for box in boxes:
        box = np.array(box).astype(np.int32)
        x, y, w, h = cv2.boundingRect(box)
        if w <= 0 or h <= 0:
            continue

        img_h, img_w = orig_img.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        if w <= 1 or h <= 1:
            continue

        boxes_location.append([x, y, w, h])

        cropped = orig_img[y:y + h, x:x + w].copy()
        if cropped.size == 0:
            continue
        box[:, 0] -= x
        box[:, 1] -= y

        box[:, 0] = np.clip(box[:, 0], 0, w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
        rectified = warp_to_rectangle(cropped, box)
        cropped_images.append(rectified)

    return cropped_images, boxes_location


def det_postprocess(
    infer_results: np.ndarray,
    orig_img: np.ndarray,
    model_height: int,
    model_width: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    heatmap = infer_results[:, :, 0].astype(np.float32)
    if float(np.max(heatmap)) > 1.5:
        heatmap = heatmap / 255.0
    return get_cropped_text_images(heatmap, orig_img, model_height, model_width)


def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    resized = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = resized
    return padded_image


def resize_with_padding(
    image: np.ndarray,
    target_height: int = 48,
    target_width: int = 320,
    pad_value: int = 128,
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left

    if image.ndim == 3:
        return cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[pad_value] * 3,
        )

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )


def draw_label(
    image: np.ndarray,
    box: Sequence[int],
    label: str,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    text_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
    padding: int = 4,
    min_font_scale: float = 0.3,
    align: str = "left",
) -> None:
    label = label.strip()
    if not label:
        return

    x, y, w, h = box
    inner_x, inner_y = x + padding, y + padding
    inner_w, inner_h = w - 2 * padding, h - 2 * padding

    font_scale = 1.0
    while font_scale >= min_font_scale:
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        if text_w <= inner_w and text_h <= inner_h:
            break
        font_scale -= 0.05

    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    if align == "left":
        text_x = inner_x
    elif align == "center":
        text_x = inner_x + max((inner_w - text_w) // 2, 0)
    elif align == "right":
        text_x = inner_x + max(inner_w - text_w, 0)
    else:
        raise ValueError(f"Unsupported alignment: {align}")

    text_y = inner_y + (inner_h + text_h) // 2
    cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)


def visualize_ocr_annotations(
    image: np.ndarray,
    boxes: Sequence[Sequence[int]],
    labels: Sequence[str],
    ocr_corrector: "OcrCorrector | None",
    text_color: Tuple[int, int, int] = (0, 0, 255),
    padding: int = 6,
    align: str = "left",
) -> np.ndarray:
    left = image.copy()
    right = image.copy()

    for (x, y, w, h), text in zip(boxes, labels):
        if not text.strip():
            continue
        cv2.rectangle(right, (x, y), (x + w, y + h), (255, 255, 255), -1)

    for (x, y, w, h), text in zip(boxes, labels):
        if not text.strip():
            continue
        if ocr_corrector:
            text = ocr_corrector.correct_text(text)
        draw_label(right, (x, y, w, h), text, text_color=text_color, padding=padding, align=align)

    return np.hstack([left, right])


def inference_result_handler(
    original_frame: np.ndarray,
    infer_results: Sequence[np.ndarray],
    boxes: Sequence[Sequence[int]],
    ocr_corrector: "OcrCorrector | None",
) -> np.ndarray:
    texts: List[str] = []
    for result in infer_results:
        pp_res = ocr_eval_postprocess(result)[0]
        texts.append(pp_res[0])
    return visualize_ocr_annotations(original_frame, boxes, texts, ocr_corrector)


class OcrCorrector:
    def __init__(self, dictionary_path: str, max_edit_distance: int = 2):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        if not os.path.exists(dictionary_path):
            raise FileNotFoundError(f"Missing dictionary file: {dictionary_path}")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def correct_text(self, text: str) -> str:
        suggestions = self.sym_spell.lookup_compound(text, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term
        return text
