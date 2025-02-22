import logging

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import norm
from scipy.spatial.distance import cdist

from src.core.config import settings

# Cấu hình logging
logging.basicConfig(
    filename="example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


class PersonID:
    def __init__(
        self,
        fullbody_embedding: np.ndarray,
        fullbody_bbox: np.ndarray,
        body_conf: np.float32,
    ):
        self.fullbody_embedding = fullbody_embedding
        self.fullbody_bbox = fullbody_bbox
        self.body_conf = body_conf
        self.ttl = settings.TIME_TO_LIVE  # Frames to live before removing from storage
        self.smooth_body = None
        self.fullbody_embeddings = None
        self.k_best = []
        self.keep = []

        self.count = {"full": 0, "top": 0, "bot": 0, "face": 0}

    def set_id(self, id: int):
        self.id = id

    def add_face_embeddings(self, face_feature, face_score):
        face_feature /= np.linalg.norm(face_feature)
        if self.face_embeddings is None:
            self.face_embeddings = face_feature
        else:
            self.count["face"] += 1
            self.face_embeddings = (
                face_feature + self.fullbody_embeddings * (self.count["face"] - 1)
            ) / self.count["face"]

    def add_fullbody_embeddings(self, body_feature, body_score):
        body_feature /= np.linalg.norm(body_feature)

        if self.fullbody_embeddings is None:
            self.body_conf = body_score
            self.first_embedding = body_feature
            self.fullbody_embeddings = body_feature
            self.last_embedding = body_feature

        # if body_score > self.body_conf:
        #     self.fullbody_embeddings = body_feature
        #     self.body_conf = body_score

        if body_score > 0.7:
            self.last_embedding = body_feature
            self.k_best.append((body_score, body_feature))

        self.k_best.sort(reverse=True, key=lambda x: x[0])
        self.k_best = self.k_best[:5]

    def add_top_embeddings(self, top_feature, top_score):
        top_feature /= np.linalg.norm(top_feature)
        if self.top_embeddings is None:
            self.top_embeddings = top_feature
        else:
            self.count["top"] += 1
            self.top_embeddings = (
                top_feature + self.top_embeddings * (self.count["top"] - 1)
            ) / self.count["top"]
        self.top_embeddings /= np.linalg.norm(self.top_embeddings)

    def add_bot_embeddings(self, bot_feature, bot_score):
        bot_feature /= np.linalg.norm(bot_feature)
        if self.bot_embeddings is None:
            self.bot_embeddings = bot_feature
        else:
            self.count["bot"] += 1
            self.bot_embeddings = (
                bot_feature + self.bot_embeddings * (self.count["bot"] - 1)
            ) / self.count["bot"]
        self.bot_embeddings /= np.linalg.norm(self.bot_embeddings)

    def get_avg_k_best_embedding(self):
        if not self.k_best:
            return None

        embeddings = np.array([x[1] for x in self.k_best])  # Lấy danh sách embeddings
        avg_embedding = np.mean(
            embeddings, axis=0
        )  # Tính trung bình cộng theo từng chiều

        return avg_embedding  # Chuẩn hóa về vector đơn vị


class PersonIDsStorage:
    def __init__(self):
        self.person_ids = []

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        return np.dot(a, b) / (norm(a) * norm(b))

    def get_person_by_id(self, id):
        for person in self.person_ids:
            if person.id == id:
                return person

    def search(
        self,
        current_person_id: PersonID,
        current_frame_id: list,
        threshold: float = 0.35,
    ):
        """
        Duyệt qua tất cả PersonID đã lưu và tìm ID gần nhất bằng cosine similarity.
        """
        most_match = None
        min_similarity = 1.0

        # Chuẩn hóa embedding hiện tại
        current_body_emb = (
            F.normalize(
                torch.tensor(current_person_id.fullbody_embedding), dim=0
            ).numpy()
            if current_person_id.fullbody_embedding is not None
            else None
        )

        for person_id in self.person_ids:
            if person_id.id not in current_frame_id:
                if person_id.fullbody_embeddings is not None:
                    person_body1 = F.normalize(
                        torch.tensor(person_id.fullbody_embeddings), dim=0
                    ).numpy()
                    per_similarity1 = np.maximum(
                        0.0,
                        cdist(
                            current_body_emb.reshape(1, -1),
                            person_body1.reshape(1, -1),
                            metric="cosine",
                        ),
                    )
                    person_body3 = F.normalize(
                        torch.tensor(person_id.first_embedding), dim=0
                    ).numpy()
                    per_similarity2 = np.maximum(
                        0.0,
                        cdist(
                            current_body_emb.reshape(1, -1),
                            person_body3.reshape(1, -1),
                            metric="cosine",
                        ),
                    )
                    person_body3 = F.normalize(
                        torch.tensor(person_id.last_embedding), dim=0
                    ).numpy()
                    per_similarity3 = np.maximum(
                        0.0,
                        cdist(
                            current_body_emb.reshape(1, -1),
                            person_body3.reshape(1, -1),
                            metric="cosine",
                        ),
                    )

                    # Thêm so sánh với trung bình `self.k_best`
                    # person_k_best_emb = person_id.get_avg_k_best_embedding()
                    # if current_body_emb is not None and person_k_best_emb is not None:
                    #     person_k_best_emb = F.normalize(torch.tensor(person_k_best_emb), dim=0).numpy()
                    #     per_similarity_kbest = np.maximum(0.0, cdist(current_body_emb.reshape(1, -1), person_k_best_emb.reshape(1, -1), metric='cosine'))
                    # else:
                    #     per_similarity_kbest = 1.0  # Nếu không có, đặt giá trị cao (ít giống)
                if len(person_id.k_best) > 0:
                    total = 0
                    for score, feature in person_id.k_best:
                        person_k_best_emb = F.normalize(
                            torch.tensor(feature), dim=0
                        ).numpy()
                        per_similarity = np.maximum(
                            0.0,
                            cdist(
                                current_body_emb.reshape(1, -1),
                                person_k_best_emb.reshape(1, -1),
                                metric="cosine",
                            ),
                        )
                        total += per_similarity
                    res = total / len(person_id.k_best)
                    logging.info(f"person_Similarity1: {per_similarity1}")
                    logging.info(f"person_Similarity2: {per_similarity2}")
                    logging.info(f"person_Similarity3: {per_similarity3}")
                    logging.info(f"person_Similarity_KBest: {res}")

                    # Lấy trung bình các similarity score
                    # similarity = (per_similarity1 + per_similarity3 + per_similarity2) / 3
                    similarity = per_similarity3
                if similarity < min_similarity and similarity < threshold:
                    most_match = person_id
                    min_similarity = similarity
                    logging.info(f"{min_similarity}, {most_match.id}")

        return most_match, min_similarity

    def add(self, person_id: PersonID):
        self.person_ids.append(person_id)

    def update_ttl(self):
        for person_id in self.person_ids:
            person_id.ttl -= 1

            if person_id.ttl <= 0:
                self.person_ids.remove(person_id)
