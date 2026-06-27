"""
util.py — Preprocessing, YOLO decode, NMS, and visualisation helpers.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, input_size: tuple[int, int] = (640, 640)):
    """
    Convert a BGR frame to a normalised RGB tensor ready for TFLite inference.

    Returns:
        tensor      – float32 ndarray of shape (1, H, W, 3)
        orig_shape  – (orig_h, orig_w) for scaling boxes back to pixel space
    """
    orig_h, orig_w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, input_size)
    tensor = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
    return tensor, (orig_h, orig_w)


# ---------------------------------------------------------------------------
# YOLO output decoding
# ---------------------------------------------------------------------------

def parse_yolo_output(
    raw_output: np.ndarray,
    conf_threshold: float = 0.30,
    orig_shape: tuple[int, int] = (640, 640),
    model_input_size: int = 640,
) -> list[list[float]]:
    """
    Decode a raw YOLOv8 TFLite output tensor.

    Expected output shape: (1, num_classes + 4, 8400)
    Column layout per anchor: [cx, cy, w, h, cls_0, cls_1, ..., cls_N]
    (YOLOv8 has NO separate objectness score — class probs are already
     filtered by the model's final sigmoid layer.)

    Returns:
        List of detections: [x1, y1, x2, y2, confidence, class_id]
        Coordinates are in the original frame's pixel space.
    """
    data = np.squeeze(raw_output)   # (num_classes+4, 8400)
    data = data.T                   # (8400, num_classes+4)

    boxes_raw   = data[:, :4]       # cx, cy, w, h  (model-space, 0-model_input_size)
    class_probs = data[:, 4:]       # (8400, num_classes)

    class_ids    = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)

    # Filter by confidence
    keep = class_scores >= conf_threshold
    boxes_raw    = boxes_raw[keep]
    class_scores = class_scores[keep]
    class_ids    = class_ids[keep]

    # Scale from model-input pixels → original frame pixels
    orig_h, orig_w = orig_shape
    scale_x = orig_w / model_input_size
    scale_y = orig_h / model_input_size

    detections: list[list[float]] = []
    for (cx, cy, bw, bh), score, cid in zip(boxes_raw, class_scores, class_ids):
        # Centre-format → corner-format, rescaled to original image
        x1 = (cx - bw / 2) * scale_x
        y1 = (cy - bh / 2) * scale_y
        x2 = (cx + bw / 2) * scale_x
        y2 = (cy + bh / 2) * scale_y
        detections.append([x1, y1, x2, y2, float(score), int(cid)])

    return detections


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def apply_nms(
    detections: list[list[float]],
    iou_threshold: float = 0.45,
) -> list[list[float]]:
    """
    Filter overlapping boxes with OpenCV's NMS.

    Input/output format: [[x1, y1, x2, y2, conf, class_id], ...]
    """
    if not detections:
        return []

    dets = np.array(detections)
    # cv2.dnn.NMSBoxes expects [x, y, w, h]
    boxes_xywh = [
        [d[0], d[1], d[2] - d[0], d[3] - d[1]]
        for d in dets
    ]
    scores = dets[:, 4].tolist()

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores,
        score_threshold=0.0,   # already filtered in parse_yolo_output
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    return [detections[i] for i in indices.flatten()]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Colour palette: each class gets a consistent colour derived from its ID
def _class_colour(class_id: int) -> tuple[int, int, int]:
    palette = [
        (0, 200, 100), (0, 120, 255), (255, 180, 0), (200, 0, 200),
        (0, 200, 200), (255, 80, 0),  (100, 0, 255), (0, 255, 180),
    ]
    return palette[class_id % len(palette)]


def draw_detections(
    frame: np.ndarray,
    detections: list[list[float]],
    names: dict[int, str],
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on *frame* (BGR, in-place).
    Returns the annotated frame.
    """
    for x1, y1, x2, y2, conf, cid in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        colour = _class_colour(cid)
        label  = names.get(cid, f"Class {cid}")
        text   = f"{label}  {conf:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, y1), colour, -1)

        # Label text
        cv2.putText(
            frame, text,
            (x1 + 2, y1 - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

    return frame
