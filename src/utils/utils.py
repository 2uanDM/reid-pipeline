import logging

import cv2
import motmetrics as mm
import numpy as np
import torch
import torchvision
from tqdm import tqdm

logging.basicConfig(
    filename="logs/example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def compute_area(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    area = int(x2 - x1) * int(y2 - y1)
    return area


def compute_iou(box1, box2):
    # Unpack coordinates
    a1, b1, a2, b2 = box1
    x1, y1, x2, y2 = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(a1, x1)
    inter_y1 = max(b1, y1)
    inter_x2 = min(a2, x2)
    inter_y2 = min(b2, y2)

    # Compute the width and height of the intersection rectangle
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # Compute the area of intersection
    inter_area = inter_width * inter_height

    # Compute the area of both boxes
    box1_area = (a2 - a1) * (b2 - b1)
    box2_area = (x2 - x1) * (y2 - y1)

    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def non_max_suppression(results, class_labels, conf_threshold=0.25, iou_threshold=0.45):
    # Extract boxes, confidences, and class IDs from YOLOv8 results
    boxes = results.boxes.xyxy
    scores = results.boxes.conf
    class_ids = results.boxes.cls

    # Initialize list for detections after NMS
    output = []

    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        if score < conf_threshold:
            continue

        class_name = class_labels[int(class_id)]
        if class_name not in class_labels:
            continue

        # Append detection to output
        output.append(
            [
                box[0].item(),
                box[1].item(),
                box[2].item(),
                box[3].item(),
                score.item(),
                class_id.item(),
            ]
        )

    # Convert to tensor
    output = torch.tensor(output)

    if len(output) == 0:
        return []

    # Apply non-max suppression
    boxes, scores = output[:, :4], output[:, 4]
    indices = torchvision.ops.nms(boxes, scores, iou_threshold)

    # Gather final detections
    output = output[indices]

    return output


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (
            (shape1[1] - shape2[1] * gain) / 2,
            (shape1[0] - shape2[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain

    coords[:, 0].clamp_(0, shape2[1])  # x1
    coords[:, 1].clamp_(0, shape2[0])  # y1
    coords[:, 2].clamp_(0, shape2[1])  # x2
    coords[:, 3].clamp_(0, shape2[0])  # y2
    return coords


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # Width, Height
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    print(f"Original shape: {shape}, Scale ratio: {r}, New pad: {pad}, w: {w}, h: {h}")

    # Ensure we have valid dimensions for resizing
    if pad[0] <= 0 or pad[1] <= 0:
        raise ValueError("Invalid padding dimensions, cannot resize image.")

    if shape[::-1] != pad:  # resize
        print(f"Resizing image from {shape[::-1]} to {pad}")
        image = cv2.resize(image, dsize=pad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT
    )  # add border
    return image, (r, r), (w, h)


def parse_ground_truth(file_path):
    ground_truths = {}
    with open(file_path, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            frame_number = int(fields[0])
            track_id = int(fields[1])
            x_min = float(fields[2])
            y_min = float(fields[3])
            width = float(fields[4])
            height = float(fields[5])

            x_max = x_min + width
            y_max = y_min + height

            detection = {
                "track_id": track_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
            if frame_number not in ground_truths:
                ground_truths[frame_number] = []
            ground_truths[frame_number].append(detection)
    return ground_truths


def evaluate_with_motmetrics(tracking_results, ground_truth_file):
    ground_truths = parse_ground_truth(ground_truth_file)
    logging.info(f"Ground truth: {ground_truths}")
    acc = mm.MOTAccumulator(auto_id=True)
    for frame_number in tqdm(tracking_results.keys()):
        detections = tracking_results[frame_number]
        if frame_number in ground_truths:
            gt_bboxes = ground_truths[frame_number]
            if len(detections) > 0 and len(gt_bboxes) > 0:
                bboxes = np.array(
                    [
                        [det["x_min"], det["y_min"], det["x_max"], det["y_max"]]
                        for det in detections
                    ]
                )
                gt_boxes = np.array(
                    [
                        [gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]]
                        for gt in gt_bboxes
                    ]
                )
                track_ids = np.array([det["track_id"] for det in detections])
                gt_ids = np.array([gt["track_id"] for gt in gt_bboxes])

                distance_matrix = mm.distances.iou_matrix(gt_boxes, bboxes, max_iou=0.5)
                acc.update(gt_ids, track_ids, distance_matrix)
            elif len(detections) == 0 and len(gt_bboxes) > 0:
                gt_ids = np.array([gt["track_id"] for gt in gt_bboxes])
                acc.update(gt_ids, [], np.zeros((len(gt_bboxes), 0)))
            else:
                track_ids = np.array([det["track_id"] for det in detections])
                acc.update([], track_ids, np.zeros((0, len(detections))))
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            "mota",
            "motp",
            "idf1",
            "num_misses",
            "num_false_positives",
            "num_switches",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
        ],
        name="acc",
    )
    return summary
