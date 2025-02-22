import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    YOLO_MODEL_PATH: str = os.path.join(os.getcwd(), "src/assets/models/detect.pt")
    SERVER_ADDR: str = "tcp://localhost:5555"
    ASYNC_MODE: bool = True
    MODEL_PATH: str = "src/assets/models/mbnv2_128x64"
    IMG_SIZE: tuple = (128, 64)
    TIME_TO_LIVE: int = 100  # 100 frames

    # Batch Processing default
    BATCH_PROCESSING_SIZE: int = 20
    THREADS: int = 10

    def model_post_init(self, _data):
        pass

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
