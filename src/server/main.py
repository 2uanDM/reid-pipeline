import json
import logging
import os
import traceback
from datetime import datetime
from typing import List

import cv2
import imagezmq
import numpy as np
from rich.console import Console

from src.core.config import settings
from src.deeputils.base_track import BaseTrack
from src.deeputils.bytetracker import BYTETracker
from src.utils.embeddings import BatchGetEmbeddingsExecutor, extract_embedding
from src.utils.logger import Logger
from src.utils.schemas import PersonID, PersonIDsStorage

logger = Logger(__name__)

console = Console()


class MainServer:
    def __init__(
        self,
        **kwargs,
    ):
        self.receiver = None
        self.video_writer = None

        self.storage = PersonIDsStorage()

        # Store the video writer for each session (source) and the tracked IDs for each session
        self.sessions_storage = {}

        # Batch processing
        self.max_batch_size = kwargs.get(
            "batch_processing_size", settings.BATCH_PROCESSING_SIZE
        )
        self.max_thread = kwargs.get("threads", settings.THREADS)

        self.batch_get_embeddings_executor = BatchGetEmbeddingsExecutor(
            max_batch_size=self.max_batch_size,
            max_thread=self.max_thread,
        )

        # Benchmark purposes
        self.log_step = [10, 40, 100, 200, 400]

    def init_receiver(self) -> bool:
        try:
            self.receiver = imagezmq.ImageHub(
                open_port="tcp://localhost:5555", REQ_REP=False
            )
        except Exception:
            console.print(
                f"[bold red]Error[/bold red] when initializing Image Receiver: {traceback.format_exc()}"
            )
            return False
        else:
            console.print(
                "[bold cyan]Image Receiver[/bold cyan] initialized [bold green]successfully[/bold green] :vampire:"
            )
            return True

    def draw_detection(self, img, bboxes, scores, ids, mask_alpha=0.3):
        height, width = img.shape[:2]
        np.random.seed(0)
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(1, 3))
        mask_img = img.copy()
        det_img = img.copy()
        size = min([height, width]) * 0.0006
        text_thickness = int(min([height, width]) * 0.001)

        for bbox, score, id_ in zip(bboxes, scores, ids):
            color = colors[0]
            bbox = np.array(bbox)
            x1, y1, x2, y2 = bbox.astype(np.int64)
            # Draw rectangle
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            caption = f"body {int(score * 100)}% ID: {id_}"
            (tw, th), _ = cv2.getTextSize(
                text=caption,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=size,
                thickness=text_thickness,
            )

            th = int(th * 1.2)
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.rectangle(mask_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            cv2.putText(
                det_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                mask_img,
                caption,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )
        return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

    def get_face_body_from_detection(self, image: np.ndarray, info: dict):
        """
        Process the detections to get body and face bounding boxes,
        filtering only those fully inside the image with a margin of 30 pixels.
        """
        height, width = image.shape[:2]  # Get image dimensions
        margin = 30  # Define the margin
        bodys = []

        for detection in info.get("detections", []):
            body = {}
            x1, y1, x2, y2 = list(map(lambda x: int(x), detection.get("bbox")))

            if (x1 < margin or x2 > width - margin) and (
                y2 > height - margin or y1 < margin
            ):
                continue
            body["bbox"] = x1, y1, x2 - x1, y2 - y1
            body["frame"] = image[y1:y2, x1:x2]

            body["score"] = detection.get("score", [])
            bodys.append(body)

        return bodys

    def reset_count(self, id):
        BaseTrack._count = id

    def extract_embeddings(self, image, bodys):
        current_person: List[PersonID] = []
        for body in bodys:
            if body.get("frame").tolist() == []:
                continue

            bbox = body.get("bbox")  # box of person

            full_body_embedding = (
                extract_embedding(
                    [
                        image[
                            (bbox[1]) : (bbox[1] + int(bbox[3])),
                            (bbox[0]) : (bbox[0] + int(bbox[2])),
                        ]
                    ],
                    async_mode=settings.ASYNC_MODE,
                )
                .detach()
                .cpu()
            )

            # Extract confidence scores
            confidence = body.get("score")

            # Create PersonID object
            person = PersonID(
                fullbody_embedding=full_body_embedding,
                fullbody_bbox=bbox,
                body_conf=confidence,
            )
            current_person.append(person)
        return current_person

    def remap_bytetracker_ids(self, bytetracker, old_id, new_id):
        """Remap a track ID in BYTETracker from old_id to new_id."""
        for track in bytetracker.tracked_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(f"Remapped BYTETracker ID from {old_id} to {new_id}")
                break

        for track in bytetracker.lost_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(f"Remapped BYTETracker ID (lost track) from {old_id} to {new_id}")
                break

        for track in bytetracker.removed_stracks:
            if track.track_id == old_id:
                track.track_id = new_id
                print(
                    f"Remapped BYTETracker ID (removed track) from {old_id} to {new_id}"
                )
                break

    def run(self):
        print(self.init_receiver())
        if not self.init_receiver():
            return

        bytetrack = BYTETracker()
        tracked_ids = np.array([], dtype=np.int32)
        output_file = "/home/quan/codes/reid-2024/app/assets/output_vid/vanh5.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        time_now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        tracking_results = {}
        sessions = {}
        previous_boxes = []
        previous_scores = []
        # Store the video writer for each session (source)
        while True:
            info, opencv_image = self.receiver.recv_image()

            info = json.loads(info)
            metadata = info.get("metadata")
            is_skipped = info.get("is_skipped")
            if metadata:
                frame_number = metadata.get("frame", [])
            else:
                print("No metadata")  # End last frame

            session_id = info.get("session_id")
            # logging.info(f"metadata: {metadata}")

            print(f"frame:{frame_number}")

            if info.get("status") == "start":
                console.print(
                    f"[bold cyan]Start[/bold cyan] processing video: {metadata.get('source')}"
                )
                console.print(f"[bold cyan]Metadata[/bold cyan]: {metadata}")
                logging.info(f"basetrack count {BaseTrack._count}")
                # Init the video writer
                time_now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

                if session_id not in sessions:
                    video_writer = cv2.VideoWriter(
                        f"trash/{time_now}-{session_id}-{os.path.basename(metadata.get('source'))}.mp4",
                        fourcc,
                        metadata.get("fps"),
                        (
                            int(metadata.get("shape")[0]),
                            int(metadata.get("shape")[1]),
                        ),
                    )
                    console.print(
                        f"Video Writer for session [bold green]{session_id}[/bold green] initialized"
                    )
                    sessions[session_id] = video_writer

            elif info.get("status") == "running":
                if session_id in sessions:
                    logging.info(f"frame_number: {frame_number + 1}")

                    video_writer = sessions.get(session_id)

                    # is_skipped=False
                    if not is_skipped:
                        bodys = self.get_face_body_from_detection(opencv_image, info)

                        current_persons = self.extract_embeddings(
                            opencv_image, bodys
                        )  # Return list of PersonID

                        boxes = np.asarray(
                            [
                                current_person.fullbody_bbox
                                for current_person in current_persons
                            ]
                        )
                        confidences = np.asarray(
                            [
                                current_person.body_conf
                                for current_person in current_persons
                            ]
                        )

                        track_bodys = np.asarray(
                            [
                                current_person.fullbody_embedding
                                for current_person in current_persons
                            ]
                        )
                        bboxes, scores, ids = bytetrack.update(
                            boxes, confidences, track_bodys
                        )

                        tracking_ids = np.array(ids).astype(np.int32)
                        new_ids = np.setdiff1d(tracking_ids, tracked_ids)
                        if new_ids.size > 0:
                            logging.info(f"{tracking_ids}")
                            logging.info(f"{new_ids}")
                        actual_new_ids = []
                        tracking_results[frame_number + 1] = []

                        wait = dict()

                        if len(bboxes) > 0:
                            persons = [
                                current_persons[i]
                                for i in range(len(current_persons))
                                if confidences[i] in scores
                            ]
                            in_frame_not_new_ids = np.setdiff1d(tracking_ids, new_ids)
                            logging.info(f"not new {in_frame_not_new_ids}")
                            for i, person in enumerate(persons):
                                person.set_id(tracking_ids[i])
                                track_id = tracking_ids[i]
                                wait[track_id] = 0
                                if track_id not in new_ids:
                                    old_person = self.storage.get_person_by_id(track_id)

                                    old_person.add_fullbody_embeddings(
                                        person.fullbody_embedding, person.body_conf
                                    )

                                elif frame_number == 0:
                                    person.set_id(tracking_ids[i])
                                    self.storage.add(person)
                                    actual_new_ids.append(tracking_ids[i])

                                    person.add_fullbody_embeddings(
                                        person.fullbody_embedding, person.body_conf
                                    )

                                elif track_id in new_ids:
                                    logging.info(f"confidence: {person.body_conf}")

                                    wait[track_id] += 1
                                    if person.body_conf > 0.7:
                                        most_match, min_similarity = (
                                            self.storage.search(
                                                person,
                                                in_frame_not_new_ids,
                                                threshold=0.25,
                                            )
                                        )
                                        if most_match:
                                            old_id = tracking_ids[i]
                                            new_id = most_match.id
                                            in_frame_not_new_ids = np.append(
                                                in_frame_not_new_ids, new_id
                                            )
                                            person.set_id(most_match.id)
                                            self.remap_bytetracker_ids(
                                                bytetrack, old_id, new_id
                                            )
                                            old_person = self.storage.get_person_by_id(
                                                new_id
                                            )

                                            old_person.add_fullbody_embeddings(
                                                person.fullbody_embedding,
                                                person.body_conf,
                                            )

                                        else:
                                            person.set_id(tracking_ids[i])
                                            self.storage.add(person)
                                            actual_new_ids.append(tracking_ids[i])
                                            in_frame_not_new_ids = np.append(
                                                in_frame_not_new_ids, tracking_ids[i]
                                            )

                                            person.add_fullbody_embeddings(
                                                person.fullbody_embedding,
                                                person.body_conf,
                                            )

                                tracking_ids[i] = person.id
                                tracking_results[frame_number + 1].append(
                                    {
                                        "track_id": tracking_ids[i],
                                        "x_min": bboxes[i][0],
                                        "y_min": bboxes[i][1],
                                        "x_max": bboxes[i][2],
                                        "y_max": bboxes[i][3],
                                        "confidence": scores[i],
                                    }
                                )
                            logging.info(f"Tracking id: {tracking_ids}")
                            tracked_ids = np.concatenate((tracked_ids, actual_new_ids))
                            logging.info(f"Tracked_id: {tracked_ids}")
                            self.reset_count(int(np.max(tracked_ids)))
                            conf_scores = np.array(scores).astype(np.float64)
                            boxes = np.array(bboxes).astype(np.float64)
                            result_img = self.draw_detection(
                                img=opencv_image,
                                bboxes=boxes,
                                scores=conf_scores,
                                ids=tracking_ids,
                            )
                            previous_boxes = boxes
                            previous_scores = conf_scores
                        else:
                            result_img = opencv_image

                    else:
                        # print(f"previous box:{previous_boxes}")
                        if previous_boxes is not None:
                            bboxes, scores, ids = bytetrack.update(
                                previous_boxes, previous_scores, track_bodys=None
                            )
                            if len(bboxes) > 0:
                                result_img = self.draw_detection(
                                    img=opencv_image,
                                    bboxes=bboxes,
                                    scores=scores,
                                    ids=ids,
                                )
                            else:
                                result_img = opencv_image
                    if self.video_writer is None:
                        h, w = opencv_image.shape[:2]
                        self.video_writer = cv2.VideoWriter(
                            output_file, fourcc, 30, (w, h)
                        )
                    text = f"Frame: {frame_number}"
                    cv2.putText(
                        result_img,
                        text,
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                    )
                    self.video_writer.write(result_img)

                else:
                    print(f"Session {session_id}")

            elif info.get("status") == "end":
                # output_file = "tracking_results.txt"
                # with open(output_file, "w") as f:
                #     for frame_number, detections in tracking_results.items():
                #         for det in detections:
                #             line = f"{frame_number},{det['track_id']},{det['x_min'] * 2},{det['y_min'] * 2},{det['x_max'] * 2 - det['x_min'] * 2},{det['y_max'] * 2 - det['y_min'] * 2}, {det['confidence']}\n"
                #             f.write(line)

                self.video_writer = sessions.pop(session_id, None)
                console.print(
                    f"[bold magenta]End processing video: {session_id}[/bold magenta]"
                )

                # Reset the video writer
                self.video_writer = None
