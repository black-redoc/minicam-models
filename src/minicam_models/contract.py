"""The model contract minicam's ONNX detection backend expects.

Everything here mirrors minicam's `app/services/detection/` (onnx_backend.py,
nms.py, coco_classes.py) and `app/routers/setup.py` as of minicam commit
5e5c028. If minicam changes any of these, this module must change with it —
see docs/ARCHITECTURE.md ("Contract invariants").

Preprocessing and decoding are copied rather than reimplemented on purpose:
validation and INT8 calibration must see exactly the tensors minicam feeds
the model at runtime (nearest-neighbor letterbox, pad 114, RGB, /255).
"""

from __future__ import annotations

import numpy as np

MODEL_INPUT_SIZE = 640
INPUT_SHAPE: tuple[int, ...] = (1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
NUM_BOX_COORDS = 4
NUM_ANCHORS = 8400
LETTERBOX_PAD_VALUE = 114
DEFAULT_IOU_THRESHOLD = 0.45

# minicam's POST /api/detection/model and /model/fetch caps.
MAX_MODEL_BYTES = 200 * 1024 * 1024
MODEL_SUFFIX = ".onnx"

# Version minicam pins in its pyproject.toml; validation runs against it.
MINICAM_ONNXRUNTIME_VERSION = "1.27.0"

COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

OUTPUT_SHAPE: tuple[int, ...] = (1, NUM_BOX_COORDS + len(COCO_CLASSES), NUM_ANCHORS)


def resize_nearest(image: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    """Resize an image with nearest-neighbor sampling, exactly like minicam."""
    height, width = image.shape[0], image.shape[1]
    row_indices = np.clip(
        (np.arange(new_height) * (height / new_height)).astype(np.int64),
        0,
        height - 1,
    )
    col_indices = np.clip(
        (np.arange(new_width) * (width / new_width)).astype(np.int64),
        0,
        width - 1,
    )
    return np.asarray(image[row_indices][:, col_indices])


def preprocess(image: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Letterbox an (H, W, 3) RGB uint8 frame into minicam's input tensor.

    Returns:
        The (1, 3, 640, 640) float32 tensor, the scale factor and the
        (pad_x, pad_y) offsets needed to map boxes back to image space.
    """
    height, width = image.shape[0], image.shape[1]
    scale = min(MODEL_INPUT_SIZE / width, MODEL_INPUT_SIZE / height)
    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = resize_nearest(image, new_width, new_height)

    pad_x = (MODEL_INPUT_SIZE - new_width) / 2
    pad_y = (MODEL_INPUT_SIZE - new_height) / 2
    canvas = np.full(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
        LETTERBOX_PAD_VALUE,
        dtype=np.uint8,
    )
    top, left = round(pad_y), round(pad_x)
    canvas[top : top + new_height, left : left + new_width] = resized

    normalized = canvas.astype(np.float32) / 255.0
    tensor = np.expand_dims(normalized.transpose(2, 0, 1), axis=0)
    return tensor, scale, pad_x, pad_y


def decode_output(
    raw_output: np.ndarray, confidence_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a raw (1, 84, 8400) tensor into xyxy boxes, scores, class ids."""
    predictions = raw_output[0].T
    box_coords = predictions[:, :NUM_BOX_COORDS]
    class_scores = predictions[:, NUM_BOX_COORDS:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    keep_mask = scores >= confidence_threshold
    cx, cy, w, h = box_coords[keep_mask].T
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes, scores[keep_mask], class_ids[keep_mask]


def box_iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU of one (4,) xyxy box against an (N, 4) array of xyxy boxes."""
    xx1 = np.maximum(box[0], others[:, 0])
    yy1 = np.maximum(box[1], others[:, 1])
    xx2 = np.minimum(box[2], others[:, 2])
    yy2 = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other_areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = area + other_areas - intersection
    return np.where(union > 0, intersection / union, 0.0)


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> np.ndarray:
    """Greedy class-agnostic NMS, same semantics as minicam's nms.py."""
    if boxes.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = order[0]
        keep.append(int(current))
        rest = order[1:]
        if rest.size == 0:
            break
        order = rest[box_iou(boxes[current], boxes[rest]) <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def postprocess(
    raw_output: np.ndarray, confidence_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode + NMS in the 640x640 letterboxed space, as minicam does."""
    boxes, scores, class_ids = decode_output(raw_output, confidence_threshold)
    keep = non_max_suppression(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep]
