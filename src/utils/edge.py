from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
from torch import Tensor


def compute_area(bbox_xyxy: Tuple[float, float, float, float]) -> int:
    """Compute area of bounding box."""
    return int((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1]))


def compute_iou(box1: Tensor, box2: Tensor) -> float:
    """
    Compute Intersection over Union (IoU) between two boxes.
    Uses vectorized operations for better performance.
    """
    # Calculate intersection coordinates
    inter_coords = torch.tensor(
        [
            max(box1[0], box2[0]),
            max(box1[1], box2[1]),
            min(box1[2], box2[2]),
            min(box1[3], box2[3]),
        ]
    )

    # Compute intersection area using max for better performance
    inter_area = max(0, inter_coords[2] - inter_coords[0]) * max(
        0, inter_coords[3] - inter_coords[1]
    )

    # Compute areas using vector operations
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def non_max_suppression(
    results: torch.Tensor,
    class_labels: List[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[torch.Tensor]:
    """
    Perform Non-Maximum Suppression on detection results.
    Optimized for memory usage and speed.
    """
    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    class_ids = results.boxes.cls

    # Pre-filter by confidence for better performance
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Convert to contiguous tensors for better memory access
    output = torch.cat(
        [boxes, scores.unsqueeze(1), class_ids.unsqueeze(1)], dim=1
    ).contiguous()

    # Filter by class labels
    valid_mask = torch.tensor(
        [int(class_labels[int(cls_id)]) in class_labels for cls_id in output[:, 5]],
        dtype=torch.bool,
    )
    output = output[valid_mask]

    if len(output) == 0:
        return []

    # Apply NMS
    indices = torchvision.ops.nms(
        boxes=output[:, :4], scores=output[:, 4], iou_threshold=iou_threshold
    )

    return output[indices].tolist()


def scale(
    coords: torch.Tensor,
    shape1: Tuple[int, int],
    shape2: Tuple[int, int],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> torch.Tensor:
    """
    Scale coordinates from one shape to another.
    Optimized with vectorized operations.
    """
    if ratio_pad is None:
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])
        pad = torch.tensor(
            [(shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2]
        )
    else:
        gain = ratio_pad[0][0]
        pad = torch.tensor(ratio_pad[1])

    # Vectorized operations for better performance
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain

    # Use torch.clamp_ for in-place operation
    coords[:, 0].clamp_(0, shape2[1])
    coords[:, 1].clamp_(0, shape2[0])
    coords[:, 2].clamp_(0, shape2[1])
    coords[:, 3].clamp_(0, shape2[0])

    return coords


def resize(
    image: np.ndarray, input_size: int
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resize image while maintaining aspect ratio.
    Returns resized image, scale ratios, and padding.
    """
    height, width = image.shape[:2]

    # Calculate scale ratio once
    ratio = min(1.0, input_size / height, input_size / width)

    # Calculate new dimensions
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))

    # Calculate padding
    pad_w = (input_size - new_width) / 2
    pad_h = (input_size - new_height) / 2

    if new_width <= 0 or new_height <= 0:
        raise ValueError("Invalid dimensions for resize operation")

    # Resize only if necessary
    if (width, height) != (new_width, new_height):
        image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    # Add border with integer padding
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return image, (ratio, ratio), (pad_w, pad_h)
