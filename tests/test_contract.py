"""Contract helpers behave like minicam's backend."""

from __future__ import annotations

import numpy as np

from minicam_models import contract


def test_coco_has_80_unique_classes_starting_with_person() -> None:
    assert len(contract.COCO_CLASSES) == 80
    assert len(set(contract.COCO_CLASSES)) == 80
    assert contract.COCO_CLASSES[0] == "person"
    assert contract.OUTPUT_SHAPE == (1, 84, 8400)


def test_preprocess_letterboxes_landscape_frame() -> None:
    frame = np.full((360, 640, 3), 200, dtype=np.uint8)
    tensor, scale, pad_x, pad_y = contract.preprocess(frame)
    assert tensor.shape == contract.INPUT_SHAPE
    assert tensor.dtype == np.float32
    assert scale == 1.0 and pad_x == 0.0 and pad_y == 140.0
    assert np.isclose(tensor[0, 0, 0, 0], 114 / 255)
    assert np.isclose(tensor[0, 0, 320, 320], 200 / 255)


def test_decode_and_nms_keep_best_of_overlapping_boxes() -> None:
    raw = np.zeros(contract.OUTPUT_SHAPE, dtype=np.float32)
    for anchor, (cx, score) in enumerate([(100, 0.9), (102, 0.8), (400, 0.7)]):
        raw[0, :4, anchor] = [cx, 100, 50, 50]
        raw[0, 4 + 16, anchor] = score  # class 16 = dog
    boxes, scores, class_ids = contract.postprocess(raw, confidence_threshold=0.5)
    assert np.allclose(scores, [0.9, 0.7])
    assert list(class_ids) == [16, 16]
    assert np.allclose(boxes[0], [75, 75, 125, 125])


def test_nms_of_empty_input_is_empty() -> None:
    keep = contract.non_max_suppression(np.empty((0, 4)), np.empty(0))
    assert keep.size == 0
