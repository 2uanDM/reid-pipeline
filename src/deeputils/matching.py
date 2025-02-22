import logging

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.deeputils import kalman_filter

logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


def merge_matches(m1, m2, shape):
    N, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(N, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))
    mask = M1 * M2
    match = mask.non_zero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(N)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = matched_cost <= thresh

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    if np.all(cost_matrix == 0):
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    cost_matrix = np.where(cost_matrix > thresh, thresh + 0.1, cost_matrix)
    if np.all(np.isinf(cost_matrix)):
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = np.array(
        [[r, c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= thresh]
    )
    unmatched_a = np.setdiff1d(
        np.arange(cost_matrix.shape[0]), matches[:, 0] if len(matches) > 0 else []
    )
    unmatched_b = np.setdiff1d(
        np.arange(cost_matrix.shape[1]), matches[:, 1] if len(matches) > 0 else []
    )

    return matches, unmatched_a, unmatched_b


def compute_iou(a_boxes, b_boxes):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes.
    :param a_boxes: list[tlbr] | np.ndarray, bounding boxes (x1, y1, x2, y2) for set A
    :param b_boxes: list[tlbr] | np.ndarray, bounding boxes (x1, y1, x2, y2) for set B
    :return: IoU matrix | np.ndarray, IoU values between each box in set A and set B
    """
    iou = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    if iou.size == 0:
        return iou

    a_boxes = np.ascontiguousarray(a_boxes, dtype=np.float32)
    b_boxes = np.ascontiguousarray(b_boxes, dtype=np.float32)

    # Get the coordinates of bounding boxes A and B
    b1_x1, b1_y1, b1_x2, b1_y2 = a_boxes.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b_boxes.T

    # Ensure coordinates are valid (i.e., x2 > x1 and y2 > y1)
    if (
        np.any(b1_x2 <= b1_x1)
        or np.any(b1_y2 <= b1_y1)
        or np.any(b2_x2 <= b2_x1)
        or np.any(b2_y2 <= b2_y1)
    ):
        print("Warning: Some bounding boxes have invalid coordinates.")

    # Compute the intersection coordinates
    inter_x1 = np.maximum(
        b1_x1[:, None], b2_x1
    )  # Top-left x-coordinate of the intersection
    inter_y1 = np.maximum(
        b1_y1[:, None], b2_y1
    )  # Top-left y-coordinate of the intersection
    inter_x2 = np.minimum(
        b1_x2[:, None], b2_x2
    )  # Bottom-right x-coordinate of the intersection
    inter_y2 = np.minimum(
        b1_y2[:, None], b2_y2
    )  # Bottom-right y-coordinate of the intersection

    # Compute the width and height of the intersection box
    inter_w = (inter_x2 - inter_x1).clip(0)  # Clip to handle non-overlapping boxes
    inter_h = (inter_y2 - inter_y1).clip(0)  # Clip to handle non-overlapping boxes

    # Compute the area of the intersection
    inter_area = inter_w * inter_h

    # Compute the area of each box in A and B
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Compute the IoU by dividing the intersection area by the union area
    union_area = box1_area[:, None] + box2_area - inter_area
    iou = inter_area / (union_area + 1e-7)  # Add epsilon to prevent division by zero

    return iou


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track._tlwh) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track._tlwh) for track in btracks]
    _ious = compute_iou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric="cosine"):
    # Initialize cost matrix with zeros
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    logging.info(f"len_track: {len(tracks)}, len_detection: {len(detections)}")
    if cost_matrix.size == 0:
        return cost_matrix

    # Get detection and track features
    detection_features = np.squeeze(
        np.asarray([track.curr_body for track in detections], dtype=np.float32)
    )
    track_features = np.squeeze(
        np.asarray([track.smooth_body for track in tracks], dtype=np.float32)
    )
    # Ensure the feature arrays are 2D
    if detection_features.ndim == 1:
        detection_features = detection_features.reshape(1, -1)
    if track_features.ndim == 1:
        track_features = track_features.reshape(1, -1)
    if len(detection_features) == 0 or len(track_features) == 0:
        return cost_matrix

    # Calculate distances using cdist
    cost_matrix = np.maximum(0.0, cdist(track_features, detection_features, metric))

    # Additional feature matching
    """track_features2 = np.squeeze(np.asarray([track.features[0] for track in tracks], dtype=np.float32))
    track_features3 = np.squeeze(np.asarray([track.features[-1] for track in tracks], dtype=np.float32))

    if track_features2.ndim == 1:
        track_features2 = track_features2.reshape(1, -1)
    if track_features3.ndim == 1:
        track_features3 = track_features3.reshape(1, -1)

    # Compute distances for additional features
    cost_matrix2 = np.maximum(0.0, cdist(track_features2, detection_features, metric))
    cost_matrix3 = np.maximum(0.0, cdist(track_features3, detection_features, metric))

    # Average the costs from the three feature sets
    for row in range(len(cost_matrix)):
        cost_matrix[row] = (cost_matrix[row] + cost_matrix2[row] + cost_matrix3[row]) / 3"""
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
