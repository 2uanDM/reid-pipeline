import json
import os
import time
import traceback
import uuid
from typing import Generator, Union

import cv2
import imagezmq
import numpy as np
import torch
from rich.console import Console
from typing_extensions import Literal
from ultralytics import YOLOv10

from src.core.config import settings
from src.utils.edge import compute_area
from src.utils.logger import Logger
from src.utils.utils import non_max_suppression

console = Console()

logger = Logger(__name__)


class BaseEdgeDevice:
    def __init__(self, source: str, server_address: str = "localhost:5555"):
        """
        Base class for edge devices

        :param source: Source of the input data. Accept:

            - Video: "video.mp4"
            - Stream: "rtsp://example.com/media.mp4"
            - Folder: "folder/" (containing videos)
        """
        self.source = source
        self.model = self._load_yolo_model()
        self.server_address = server_address

    def log(
        self,
        message: str,
        type: Literal["info", "warning", "error"] = "info",
    ):
        if type == "info":
            logger.info(message)
            console.print(f"[bold green]Info[/bold green]: {message}")
        elif type == "warning":
            logger.warning(message)
            console.print(f"[bold yellow]Warning[/bold yellow]: {message}")
        elif type == "error":
            logger.error(message)
            console.print(f"[bold red]Error[/bold red]: {message}")
        else:
            raise ValueError(
                "Invalid log type. Must be one of 'info', 'warning', 'error'"
            )

    def _load_yolo_model(self) -> YOLOv10:
        try:
            self.log(f"Loading detection model: {settings.YOLO_MODEL_PATH}")
            return YOLOv10(settings.YOLO_MODEL_PATH)
        except Exception as e:
            self.log(
                f"Error when loading detection model: {traceback.format_exc()}",
                type="error",
            )
            raise e

    def _read_video(self) -> Generator:
        try:
            cap = cv2.VideoCapture(self.source)
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                yield cap, frame
        except Exception as e:
            self.log(
                message=f"Error when open video file: {traceback.format_exc()}",
                type="error",
            )
            raise e
        finally:
            cap.release()

    def _read_rtsp(self) -> Generator:
        NotImplementedError("Source RTSP stream has not been supported yet")

    def _read_folder(self) -> Generator:
        raise NotImplementedError("Source folder has not been supported yet")

    def read_source(self) -> Generator:
        """
        Convert the input source to frames, then yield them
        """

        if not self.source or not isinstance(self.source, str):
            raise ValueError("Invalid source. Source must be a string")

        if self.source.startswith("rtsp://"):
            return self._read_rtsp()
        elif os.path.isfile(self.source):
            return self._read_video()
        elif os.path.isdir(self.source):
            return self._read_folder()
        else:
            raise ValueError(
                f"Invalid source: {self.source}. Must be a valid video path, stream url, or folder path."
            )

    def _post_process_detect_result(
        self,
        results: YOLOv10,
        height: int,
        width: int,
    ) -> list:
        """
        To enhance tracking performance, we need to ensure that both the body and face are appearing and disappearing at the same time.

        This function will help us to filter out the bounding boxes that are not in pair.

        Args:
            results: YOLOv10 results object

        Returns:
            list of bounding boxes, scores, class_ids, names
        """
        if results:
            detections = non_max_suppression(results, ["body", "face"], 0.2, 0.7)
            if detections.numel() > 0:
                boxes = detections[:, :4].cpu().numpy()

                self.log(f"Number of boxes:{len(boxes)}")

                total_area = sum([compute_area(box) for box in boxes])
                num_boxes = len(boxes)
                mean_area = total_area / num_boxes
                sum_criterion = (
                    (mean_area / (height * width)) + (0.1 * num_boxes)
                ) * 0.1
                criterion = sum_criterion
                boxes = boxes.astype(np.int32)
                confidences = detections[:, 4].cpu().numpy()
                class_ids = detections[:, 5].cpu().numpy()
                frame_limit = 3

        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int32)
            criterion = 0.01
            frame_limit = 15

        return boxes, confidences, class_ids, criterion, frame_limit

    def detect(self, idx: int, frame: np.ndarray):
        try:
            print(f"Frame shape: {frame.shape}")
            height, width = frame.shape[:2]

            # Perform inference
            with torch.no_grad():
                results_body = self.model.predict(frame, conf=0.4)[0]

            # Post-processing of detection results
            bboxes_xyxy, scores, class_ids, criterion, frame_limit = (
                self._post_process_detect_result(results_body, height, width)
            )

            output = []
            bodys = []
            faces = []

            # Separate bodies and faces based on class ID
            if len(bboxes_xyxy) > 0:
                bboxes = bboxes_xyxy.tolist()
                scores = scores.tolist()
                class_ids = class_ids.tolist()

                for bbox, score, class_id in zip(bboxes, scores, class_ids):
                    if class_id == 0:  # Assuming class_id 0 is for bodies
                        bodys.append(
                            {
                                "bbox": bbox,
                                "score": score,
                                "class_id": int(class_id),
                                "clothes": [],
                            }
                        )
                    elif class_id == 1:  # Assuming class_id 1 is for faces
                        faces.append(
                            {
                                "bbox": bbox,
                                "score": score,
                                "class_id": int(class_id),
                            }
                        )

                # Check if faces are inside bodies
                for face in faces:
                    face_bbox = face["bbox"]
                    for body in bodys:
                        body_bbox = body["bbox"]
                        if self.check_inside(face_bbox, body_bbox):
                            body["face"] = face_bbox
                            body["face_conf"] = face["score"]
                            break

                output.extend(bodys)

            self.log(f"Processing frame {idx + 1}")

            return output, criterion, frame_limit
        except Exception:
            console.print(
                f"[bold red]Error[/bold red] when detecting frame {idx}: {traceback.format_exc()}"
            )

    @staticmethod
    def check_inside(box_a: list, box_b: list) -> bool:
        x1_a, y1_a, x2_a, y2_a = box_a
        x1_b, y1_b, x2_b, y2_b = box_b

        return x1_a >= x1_b and y1_a >= y1_b and x2_a <= x2_b and y2_a <= y2_b

    def okay_to_run_detect(
        self,
        cur_frame: np.ndarray,
        prev_frame: Union[np.ndarray, None],
        frame_count: int,
        criterion: int,
        frame_limit: int,
    ) -> bool:
        """
        Check if it's okay to run detection on the current frame

        :params:
            cur_frame: current frame
            prev_frame: previous frame
            frame_count: number of frames since last detection

        :return:
            True if it's okay to run detection, False otherwise
        """
        # Convert to gray
        height, width = cur_frame.shape[:2]
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the percentage of changed pixels between the current frame and the previous frame
        if prev_frame is not None:
            frame_diff = cv2.absdiff(cur_gray, prev_frame)
            _, thresh = cv2.threshold(frame_diff, 150, 255, cv2.THRESH_BINARY)
            non_zero_count = np.count_nonzero(thresh)
            total_pixels = cur_gray.shape[0] * cur_gray.shape[1]

            change_pct = (non_zero_count / total_pixels) * 100
        else:
            change_pct = 0
            frame_count = 0

        if change_pct > criterion or frame_count == frame_limit or prev_frame is None:
            return True
        else:
            return False

    def init_sender(self) -> bool:
        try:
            self.sender = imagezmq.ImageSender(
                connect_to=settings.SERVER_ADDR, REQ_REP=False
            )
        except Exception:
            console.print(
                f"[bold red]Error[/bold red] when initializing Image Sender: {traceback.format_exc()}"
            )
            return False
        else:
            console.print(
                "[bold cyan]Image Sender[/bold cyan] initialized [bold green]successfully[/bold green] :vampire:"
            )
            return True

    def get_start_stop_msg(
        self,
        cap: cv2.VideoCapture,
        is_start: bool,
        session_id: str,
        summary: Union[dict, None] = None,
    ) -> tuple:
        if is_start:
            msg = json.dumps(
                {
                    "session_id": session_id,
                    "status": "start",
                    "metadata": {
                        "source": self.source,
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "length": cap.get(cv2.CAP_PROP_FRAME_COUNT),
                        "shape": (
                            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        ),
                    },
                }
            )

            return msg, np.zeros((640, 480, 3), np.uint8)
        else:
            return json.dumps(
                {"session_id": session_id, "status": "end", "summary": summary}
            ), np.zeros((640, 480, 3), np.uint8)

    def run(self):
        if not self.init_sender():
            return

        output_video_path = "output/output_vid1.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        cap = cv2.VideoCapture(self.source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out_video = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

        # Warmup the server
        time.sleep(2)

        prev_frame = None
        frame_count = 0
        skipped_frames = 0
        criterion = 0.4
        frame_limit = 5
        time_start = time.time()
        session_id = uuid.uuid4().hex

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.sender.send_image(
            *self.get_start_stop_msg(cap, is_start=True, session_id=session_id)
        )

        # Warmup the server
        time.sleep(2)

        for idx, (cap, frame) in enumerate(self.read_source()):
            metadata = {
                "frame": idx,
                "source": self.source,
                "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
            }
            if self.okay_to_run_detect(
                frame, prev_frame, frame_count, criterion, frame_limit
            ):
                output, criterion, frame_limit = self.detect(idx, frame)

                frame_count += 1
                msg = json.dumps(
                    {
                        "session_id": session_id,
                        "status": "running",
                        "metadata": metadata,
                        "is_skipped": False,
                        "detections": output,
                    }
                )
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_count = 0
                self.sender.send_image(msg, frame)
                out_video.write(frame)

            else:
                frame_count += 1
                skipped_frames += 1
                msg = json.dumps(
                    {
                        "session_id": session_id,
                        "status": "running",
                        "metadata": metadata,
                        "is_skipped": True,
                        "detections": [],
                    }
                )

                self.sender.send_image(msg, frame)
                out_video.write(frame)

        out_video.release()
        print(time.time() - time_start)

        self.sender.send_image(
            *self.get_start_stop_msg(
                cap,
                is_start=False,
                session_id=session_id,
                summary={
                    "time_elapsed": time.time() - time_start,
                    "skipped_frames": skipped_frames,
                    "total_frames": int(total_frames),
                },
            )
        )

        cap.release()
