from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    def model_post_init(self, _data):
        pass

    model_config = {"env_file": ".env", "case_sensitive": True}


settings = Settings()
