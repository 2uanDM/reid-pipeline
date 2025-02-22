import motmetrics as mm
import numpy as np
import tqdm


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
