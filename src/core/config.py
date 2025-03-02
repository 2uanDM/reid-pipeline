import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    YOLO_MODEL_PATH: str = os.path.join(os.getcwd(), "src/assets/models/detect.onnx")
    ASYNC_MODE: bool = False
    MODEL_PATH: str = "src/assets/models/mbnv2_128x64"
    IMG_SIZE: tuple = (128, 64)
    TIME_TO_LIVE: int = 100  # 100 frames

    # Batch Processing default
    BATCH_PROCESSING_SIZE: int = 10
    THREADS: int = 10

    # Add these settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PREFIX: str = "personid:"
    FRAME_TTL_TO_SECONDS: int = 1  # Conversion factor from frame TTL to seconds

    def model_post_init(self, _data):
        pass

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
