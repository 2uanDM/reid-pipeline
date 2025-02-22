import logging
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from src.deeputils import matching
from src.deeputils.base_track import BaseTrack, TrackState
from src.deeputils.kalman_filter import KalmanFilter

logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, body_feature, buffer_size=30):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.smooth_body = None
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.total_frame = 0
        self.update_bodys(body_feature, score)

    def update_bodys(self, body_feature, score):
        self.total_frame += 1
        body_feature /= np.linalg.norm(body_feature)
        self.curr_body = body_feature

        if self.smooth_body is None:
            self.smooth_body = body_feature
        else:
            self.smooth_body = (
                self.smooth_body * (1 - self.alpha) + self.alpha * body_feature
            )
        self.features.append(body_feature)
        self.smooth_body /= np.linalg.norm(self.smooth_body)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track._tlwh)
        )
        self.update_bodys(new_track.curr_body, new_track.score)
        self._tlwh = new_track._tlwh
        # self.update_faces(new_track.curr_face)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, update_feature=True):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track._tlwh
        self._tlwh = new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        if update_feature:
            # self.update_faces(new_track.curr_face)
            self.update_bodys(new_track.curr_body, new_track.score)

    @staticmethod
    def reset_id():
        STrack.count = 0

    @property
    def tlwh(self):
        # Convert x_center,y_center,a,h to x1,y1,w,h
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        # Convert x1,y1,w,h to x1,y1,x2,y2
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return "OT_{}_({}-{})".format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker:
    def __init__(self, frame_rate=30):
        self.tracked_stracks = []  # tracked in latest frame
        self.lost_stracks = []  # lost in latest frame
        self.removed_stracks = []  # lost has been removed
        self.frame_id = 0
        self.low_thresh = 0.35  # Threshold value for matching
        self.track_thresh = 0.7  # High thresh
        self.det_thresh = 0.8
        self.buffer_size = int(frame_rate / 30.0 * 30)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.output_stracks = []

    def update(self, bboxes, scores, track_bodys):
        if track_bodys is not None:
            self.frame_id += 1
            activated_stracks = []
            refind_stracks = []
            lost_stracks = []
            removed_stracks = []
            if track_bodys.size > 0:  # Ensure track_bodys is not empty
                track_bodys_tensor = torch.tensor(track_bodys, dtype=torch.float32)

                # Check if track_bodys_tensor has at least two dimensions for normalization
                if track_bodys_tensor.ndim > 1:
                    track_body_feature = (
                        F.normalize(track_bodys_tensor, dim=1).cpu().numpy()
                    )
                else:
                    track_body_feature = (
                        F.normalize(track_bodys_tensor, dim=0).cpu().numpy()
                    )
            else:
                track_body_feature = np.array([])
            remain_inds = scores > self.track_thresh  # High confidence
            indices_low = scores > self.low_thresh
            indices_high = scores < self.track_thresh
            indices_second = np.logical_and(indices_low, indices_high)  # Low confidence

            boxes_keep = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            id_feature = track_body_feature[remain_inds]
            boxes_second = bboxes[indices_second]
            scores_second = scores[indices_second]
            id_feature_second = track_body_feature[indices_second]

            if len(boxes_keep) > 0:
                detections = [
                    STrack(tlwh, s, f)
                    for (tlwh, s, f) in zip(boxes_keep, scores_keep, id_feature)
                ]
            else:
                detections = []
            """Step 1: Add newly detected tracklets to tracked_stracks"""
            unconfirmed = []
            tracked_stracks = []
            for track in self.tracked_stracks:
                if not track.is_activated:
                    unconfirmed.append(track)
                else:
                    tracked_stracks.append(track)
            """Step2: First association, with high score detection boxes"""
            strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
            logging.info(f"strack_pool: {strack_pool}")
            STrack.multi_predict(strack_pool)
            dists = matching.embedding_distance(strack_pool, detections)
            logging.info(f"Dist0_0: {dists}")
            dists = matching.gate_cost_matrix(
                self.kalman_filter, dists, strack_pool, detections
            )
            dists = matching.fuse_motion(
                self.kalman_filter, dists, strack_pool, detections
            )
            logging.info(f"Dist0: {dists}")

            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=0.3
            )
            for tracked_i, box_i in matches:
                track = strack_pool[tracked_i]
                box = detections[box_i]
                if track.state == TrackState.Tracked:
                    track.update(box, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(box, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            """Step 3: Second association, with IOU"""
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [
                strack_pool[i]
                for i in u_track
                if strack_pool[i].state == TrackState.Tracked
            ]
            dists = matching.iou_distance(r_tracked_stracks, detections)

            logging.info(f"dists2: {dists}")
            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=0.7
            )
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)

                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            """Step 3.5: Second association with IOU"""
            if len(boxes_second) > 0:
                detections_second = [
                    STrack(tlwh, s, f)
                    for (tlwh, s, f) in zip(
                        boxes_second, scores_second, id_feature_second
                    )
                ]
            else:
                detections_second = []
            second_tracked_stracks = [
                r_tracked_stracks[i]
                for i in u_track
                if r_tracked_stracks[i].state == TrackState.Tracked
            ]
            dists = matching.iou_distance(second_tracked_stracks, detections_second)
            logging.info(f"dists3: {dists}")
            matches, u_track_second, u_detection_second = matching.linear_assignment(
                dists, thresh=0.75
            )
            for itracked, idet in matches:
                track = second_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    logging.info(track.track_id)
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                    logging.info("Update duoc")
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                    logging.info("Tim lai duoc")

            for it in u_track_second:
                track = second_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
                    logging.info("Mark this track as lost")

            """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
            detections = [detections[i] for i in u_detection]

            dists = matching.iou_distance(unconfirmed, detections)
            logging.info(f"dists4: {dists}")
            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=0.7
            )
            for itracked, idet in matches:
                unconfirmed[itracked].update(detections[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])

            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
                logging.info(f"Remove this track: {removed_stracks}")

            """Step 4: Init new stracks"""
            for inew in u_detection:
                track = detections[inew]
                if track.score < self.det_thresh:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)
                logging.info("Activate new track_id")
            """Step 5: Update state"""
            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
                    logging.info("Remove this track because of exceed frame")

            self.tracked_stracks = [
                t for t in self.tracked_stracks if t.state == TrackState.Tracked
            ]
            self.tracked_stracks = joint_stracks(
                self.tracked_stracks, activated_stracks
            )
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)

            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
                self.tracked_stracks, self.lost_stracks
            )
            self.output_stracks = [
                track for track in self.tracked_stracks if track.is_activated
            ]
            bboxes = []
            scores = []
            ids = []
            logging.info(self.tracked_stracks)
            for track in self.output_stracks:
                if track.is_activated:
                    track_bbox = track.tlbr
                    bboxes.append(
                        [
                            max(0, track_bbox[0]),
                            max(0, track_bbox[1]),
                            track_bbox[2],
                            track_bbox[3],
                        ]
                    )
                    scores.append(track.score)
                    ids.append(track.track_id)
            return bboxes, scores, ids
        else:
            bboxes = []
            scores = []
            ids = []
            self.frame_id += 1
            for track in self.output_stracks:
                if track.is_activated:
                    STrack.predict(track)
                    track_bbox = track.tlbr
                    bboxes.append(
                        [
                            max(0, track_bbox[0]),
                            max(0, track_bbox[1]),
                            track_bbox[2],
                            track_bbox[3],
                        ]
                    )
                    scores.append(track.score)
                    ids.append(track.track_id)
            return bboxes, scores, ids

    @staticmethod
    def reset_id():
        STrack.reset_id()


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t

    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0) and stracks[tid].start_frame - t.end_frame <= 30:
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb


def remove_fp_stracks(stracksa, n_frame=10):
    remain = []
    for t in stracksa:
        score_5 = t.score_list[-n_frame:]
        score_5 = np.array(score_5, dtype=np.float32)
        index = score_5 < 0.45
        num = np.sum(index)
        if num < n_frame:
            remain.append(t)
    return remain
