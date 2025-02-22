import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    YOLO_MODEL_PATH: str = os.path.join(os.getcwd(), "data/weights/yolov10n_body.pt")
    SERVER_ADDR: str = "tcp://localhost:5555"
    ASYNC_MODE: bool = True
    MODEL_PATH: str = "src/assets/models/mbnv2_128x64"
    IMG_SIZE: tuple = (128, 64)
    TIME_TO_LIVE: int = 100  # 100 frames

    def model_post_init(self, _data):
        pass

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
