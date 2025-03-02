import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import redis
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from src.core.config import settings

# Cấu hình logging
logging.basicConfig(
    filename="logs/example.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


class PersonID:
    def __init__(
        self,
        fullbody_embedding: np.ndarray,
        fullbody_bbox: np.ndarray,
        body_conf: np.float32,
    ):
        self.id = None
        self.fullbody_embedding = fullbody_embedding
        self.fullbody_bbox = fullbody_bbox
        self.body_conf = body_conf
        self.ttl = settings.TIME_TO_LIVE  # Frames to live before removing from storage
        self.smooth_body = None
        self.fullbody_embeddings = None
        self.first_embedding = None
        self.last_embedding = None
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert PersonID object to dictionary for Redis storage"""

        # Helper function to handle any array-like structure safely
        def safe_convert(obj):
            if obj is None:
                return None
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):  # Add this to handle PyTorch tensors
                return obj.cpu().detach().numpy().tolist()
            elif isinstance(obj, (list, tuple)):
                return list(obj)
            # Handle NumPy scalar types - using correct type names
            elif isinstance(obj, np.integer):  # This covers all integer types
                return int(obj)
            elif isinstance(obj, np.floating):  # This covers all float types
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj

        return {
            "id": safe_convert(self.id),
            "fullbody_embedding": safe_convert(self.fullbody_embedding),
            "fullbody_bbox": safe_convert(self.fullbody_bbox),
            "body_conf": float(self.body_conf) if self.body_conf is not None else None,
            "ttl": safe_convert(self.ttl),
            "fullbody_embeddings": safe_convert(self.fullbody_embeddings),
            "first_embedding": safe_convert(self.first_embedding),
            "last_embedding": safe_convert(self.last_embedding),
            "k_best": [
                (float(score), safe_convert(feature)) for score, feature in self.k_best
            ],
            "count": {k: safe_convert(v) for k, v in self.count.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonID":
        """Create PersonID object from dictionary"""

        # Helper function to convert lists to NumPy arrays if needed
        def safe_convert_to_np(obj):
            if obj is None:
                return None
            elif isinstance(obj, list):
                return np.array(obj)
            return obj

        # Create instance with required fields
        instance = cls(
            fullbody_embedding=safe_convert_to_np(data["fullbody_embedding"]),
            fullbody_bbox=safe_convert_to_np(data["fullbody_bbox"]),
            body_conf=np.float32(data["body_conf"]) if data["body_conf"] else None,
        )

        # Set all other fields
        instance.id = data["id"]
        instance.ttl = data["ttl"]
        instance.fullbody_embeddings = safe_convert_to_np(data["fullbody_embeddings"])
        instance.first_embedding = safe_convert_to_np(data["first_embedding"])
        instance.last_embedding = safe_convert_to_np(data["last_embedding"])
        instance.k_best = [
            (float(score), safe_convert_to_np(feature))
            for score, feature in data["k_best"]
        ]
        instance.count = data["count"]

        return instance


class RedisPersonIDsStorage:
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_prefix: str = "personid:",
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=False,
        )
        self.prefix = redis_prefix
        # Set to track active IDs in the current frame
        self.current_frame_ids = set()

    def _person_key(self, id: int) -> str:
        """Generate Redis key for person ID"""
        return f"{self.prefix}{id}"

    def _get_all_keys(self) -> List[str]:
        """Get all person keys in Redis"""
        return [key.decode() for key in self.redis_client.keys(f"{self.prefix}*")]

    def _get_all_ids(self) -> List[int]:
        """Get all person IDs in Redis"""
        keys = self._get_all_keys()
        return [int(key.replace(self.prefix, "")) for key in keys]

    def get_person_by_id(self, id: int) -> Optional[PersonID]:
        """Get person by ID from Redis"""
        key = self._person_key(id)
        data = self.redis_client.get(key)
        if data:
            person_dict = json.loads(data.decode())
            return PersonID.from_dict(person_dict)
        return None

    def add(self, person_id: PersonID):
        """Add person to Redis with TTL"""
        if person_id.id is None:
            raise ValueError("Person ID must be set before adding to storage")

        key = self._person_key(person_id.id)

        try:
            person_dict = person_id.to_dict()
            json_data = json.dumps(person_dict)
            self.redis_client.set(key, json_data)
            # Set expiration time based on TTL (converted to seconds)
            self.redis_client.expire(key, person_id.ttl * settings.FRAME_TTL_TO_SECONDS)
        except Exception as e:
            logging.error(f"Error serializing person ID {person_id.id}: {str(e)}")
            logging.error(f"Person data: {person_id.__dict__}")
            raise

    def update(self, person_id: PersonID):
        """Update person in Redis"""
        self.add(person_id)  # Use add method for update as well

    def update_ttl(self):
        """
        Update TTL for all persons in Redis
        Note: Redis handles TTL automatically, but we need to
        decrement our internal TTL counter as well
        """
        for key in self._get_all_keys():
            data = self.redis_client.get(key)
            if data:
                person_dict = json.loads(data.decode())
                person_dict["ttl"] -= 1

                if person_dict["ttl"] <= 0:
                    # Remove from Redis if TTL is expired
                    self.redis_client.delete(key)
                else:
                    # Update the TTL in Redis
                    self.redis_client.set(key, json.dumps(person_dict))

    def search(
        self,
        current_person_id: PersonID,
        current_frame_id: list,
        threshold: float = 0.35,
    ) -> Tuple[Optional[PersonID], float]:
        """
        Search for the most similar person in Redis using cosine similarity.
        Keeps the same algorithm from the original implementation.
        """
        most_match = None
        min_similarity = 1.0

        # Normalize current embedding
        current_body_emb = (
            F.normalize(
                torch.tensor(current_person_id.fullbody_embedding), dim=0
            ).numpy()
            if current_person_id.fullbody_embedding is not None
            else None
        )

        # If there's no embedding, we can't compare
        if current_body_emb is None:
            return None, 1.0

        # Check all person IDs in Redis
        for person_key in self._get_all_keys():
            person_id_value = person_key.replace(self.prefix, "")

            if int(person_id_value) not in current_frame_id:
                # Get person data from Redis
                data = self.redis_client.get(person_key)
                if not data:
                    continue

                person_dict = json.loads(data.decode())
                person_id = PersonID.from_dict(person_dict)

                # Initialize similarity as highest possible value (worst match)
                similarity = 1.0

                if person_id.fullbody_embeddings is not None:
                    # Calculate similarities using the same algorithm
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
                    )[0][0]  # Extract the scalar value

                    person_body2 = F.normalize(
                        torch.tensor(person_id.first_embedding), dim=0
                    ).numpy()
                    per_similarity2 = np.maximum(
                        0.0,
                        cdist(
                            current_body_emb.reshape(1, -1),
                            person_body2.reshape(1, -1),
                            metric="cosine",
                        ),
                    )[0][0]  # Extract the scalar value

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
                    )[0][0]  # Extract the scalar value

                    # Set similarity from per_similarity3
                    similarity = per_similarity3

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
                        )[0][0]  # Extract the scalar value
                        total += per_similarity
                    res = total / len(person_id.k_best)
                    logging.info(f"person_Similarity1: {per_similarity1}")
                    logging.info(f"person_Similarity2: {per_similarity2}")
                    logging.info(f"person_Similarity3: {per_similarity3}")
                    logging.info(f"person_Similarity_KBest: {res}")

                    # Use the same similarity calculation as original
                    similarity = per_similarity3

                # Now we can compare with the threshold
                if similarity < min_similarity and similarity < threshold:
                    most_match = person_id
                    min_similarity = similarity
                    logging.info(f"{min_similarity}, {most_match.id}")

        return most_match, min_similarity

    def clear(self):
        """Clear all person IDs from Redis"""
        keys = self._get_all_keys()
        if keys:
            self.redis_client.delete(*keys)
