import json
import logging
import os
import time
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
from src.utils.benchmark import BenchmarkMultiThread
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
        if kwargs.get("benchmark"):
            self.benchmark = True
        else:
            self.benchmark = False
            
        self.log_step = [10, 40, 100, 200, 400]

    def init_receiver(self) -> bool:
        try:
            self.receiver = imagezmq.ImageHub(
                open_port="tcp://localhost:5555", REQ_REP=False
            )
        except Exception:
            console.log(
                f"[bold red]Error[/bold red] when initializing Image Receiver: {traceback.format_exc()}"
            )
            return False
        else:
            console.log(
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

    @property
    def current_time_str(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def handle_start_frame(
        self,
        fourcc: int,
        metadata: dict,
        session_id: str,
    ):
        """
        Handle the start frame of a session

        Args:
            fourcc: The fourcc codec for the video writer
            metadata: The metadata of the session
            session_id: The ID of the session
            session_storage: The storage for the current session

        Each session will look like this:
        ```json
        {
            "video_writer": cv2.VideoWriter,
            "byte_tracker": BYTETracker,
            "save_dir": str,
            "tracked_ids": np.array,
            "tracking_results": dict,
        }
        ```
        """
        console.log(
            f"[bold cyan]Start[/bold cyan] processing video: {metadata.get('source')}"
        )
        console.log(
            f"[bold cyan]Metadata[/bold cyan]: {json.dumps(metadata, indent=2)}"
        )
        logging.info(f"Base track count {BaseTrack._count}")

        # Init the video writer
        if session_id not in self.sessions_storage:
            save_dir = f"trash/{self.current_time_str}_{session_id}"
            os.makedirs(save_dir, exist_ok=True)

            save_path = f"{save_dir}/{os.path.basename(metadata.get('source'))}.mp4"

            self.sessions_storage[session_id] = {
                "video_writer": cv2.VideoWriter(
                    save_path,
                    fourcc,
                    metadata.get("fps"),
                    (
                        int(metadata.get("shape")[0]),
                        int(metadata.get("shape")[1]),
                    ),
                ),
                "byte_tracker": BYTETracker(),
                "save_dir": save_dir,
                "tracked_ids": np.array([], dtype=np.int32),
                "tracking_results": {},
                "start_time": time.perf_counter(),
                "previous_boxes": None,
                "previous_scores": None,
            }

            # Start logging the benchmark data
            if self.benchmark:
                self.sessions_storage[session_id]["benchmark"] = {
                    "executor": BenchmarkMultiThread(),
                    "data": {},
                }
                self.sessions_storage[session_id]["benchmark"][
                    "executor"
                ].start_logging()

            console.log(
                f"Save dir for session [bold green]{session_id}[/bold green]: {save_dir}"
            )

    def run(self):
        if not self.init_receiver():
            return

        bytetrack = BYTETracker()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        tracked_ids = np.array([], dtype=np.int32)
        tracking_results = {}
        previous_boxes = []
        previous_scores = []

        while True:
            info, opencv_image = self.receiver.recv_image()
            info = json.loads(info)

            detections = info.get("detections")
            metadata = info.get("metadata")
            session_id = info.get("session_id")
            is_skipped = info.get("is_skipped")

            if metadata:
                frame_number = metadata.get("frame", [])
            else:
                console.log("[bold red]No Metadata[/bold red]")

            if info.get("status") == "start":
                self.step_ids = 0

                self.handle_start_frame(
                    fourcc=fourcc,
                    metadata=metadata,
                    session_id=session_id,
                )

            elif info.get("status") == "running":
                self.step_ids = 0

                if self.benchmark and self.step_ids in self.log_step:
                    executor: BenchmarkMultiThread = self.sessions_storage[session_id][
                        "benchmark"
                    ]["executor"]["executor"]

                    data = executor.calculate_averages()

                    console.log(f"Usage at step {self.step_ids}: {data}")

                    # Store the benchmark data
                    self.sessions_storage[session_id]["benchmark"]["data"][
                        self.step_ids
                    ] = data

                # BATCH PROCESSING MECHANISM - Accumulate frames until the batch is full
                console.log(
                    f"[bold yellow][ACCUMULATING][/bold yellow] frame:{frame_number} of session [bold cyan]{session_id}[/bold cyan]"
                )

                if opencv_image is not None and (
                    not self.batch_get_embeddings_executor.queue_is_full
                ):
                    self.batch_get_embeddings_executor.add(
                        image=opencv_image,
                        detections=detections,
                        metadata=metadata,
                        session_id=session_id,
                    )
                    continue  # Continue accumulating until the batch is full

                # Process the batch to get embeddings
                # Here batch_result_frames is a list of results for each frame in the batch also sorted by the original order when they are added
                console.log("[bold green]Processing batch[/bold green]")

                _time = time.time()
                batch_result_frames = self.batch_get_embeddings_executor.process()
                console.print(f"Processing time: {time.time() - _time}s")

                # Continue the flow with the processed frames
                for frame_result in batch_result_frames:
                    current_persons: List[PersonID] = frame_result.get("persons", [])
                    metadata = frame_result.get("metadata", {})
                    session_id: str = frame_result.get("session_id")
                    frame_number: int = metadata.get("frame")
                    # original_image: np.ndarray = frame_result.get("original_image")
                    is_skipped: bool = frame_result.get("is_skipped")
                    video_writer: cv2.VideoWriter = self.sessions_storage[session_id][
                        "video_writer"
                    ]

                    # Detach all the embedding of current_persons
                    for person in current_persons:
                        person.fullbody_embedding = (
                            person.fullbody_embedding.detach().cpu().numpy()
                        )

                    if not is_skipped:
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
                    video_writer.write(result_img)

                else:
                    print(f"Session {session_id}")

            elif info.get("status") == "end":
                # Get session data
                session_data = self.sessions_storage.pop(session_id, None)

                if session_data:
                    # Clean up video writer
                    if session_data["video_writer"]:
                        session_data["video_writer"].release()

                    # Calculate and log processing time
                    processing_time = time.perf_counter() - session_data["start_time"]
                    console.log(
                        f"[bold magenta]Finished processing session: {session_id}[/bold magenta]"
                        f"\nTotal processing time: {processing_time:.2f} seconds"
                    )

                    # Log benchmark data if enabled
                    if self.benchmark and "benchmark" in session_data:
                        benchmark_data = session_data["benchmark"]["data"]
                        console.log(
                            f"Benchmark results: {json.dumps(benchmark_data, indent=2)}"
                        )

                    # Clear session resources
                    del session_data
                else:
                    console.log(
                        f"[bold red]Warning:[/bold red] No session data found for {session_id}"
                    )
