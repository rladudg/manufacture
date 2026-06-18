from pathlib import Path
from typing import Literal
import os
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YOLO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # 서버
    host: str = "0.0.0.0"
    port: int = 5001
    workers: int = 1

    # 모델
    model_path: str = "service/models/cam1_v1_d1_20260424.pt"
    model_version: str = "cam1_v1_d1_20260424"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    img_size: int = 640
    device: Literal["cpu", "cuda", "mps"] = "cpu"

    # 추론 제한
    max_image_bytes: int = 10 * 1024 * 1024  # 10MB
    inference_timeout_ms: int = 1500
    max_concurrent_inferences: int = 1

    # 로깅
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "json"
    log_dir: str = "logs"

    # Active Learning
    active_learning_enabled: bool = False
    active_learning_dir: str = "active_learning/captures"
    confidence_low_threshold: float = 0.70

    # Roboflow
    roboflow_api_key: str = ""
    roboflow_workspace: str = ""
    roboflow_project: str = ""

    env: Literal["dev", "prod"] = "dev"

    @classmethod
    def load(cls) -> "ServerSettings":
        env = os.getenv("YOLO_ENV", "dev")
        config_dir = Path(__file__).parent

        merged: dict = {}
        for fname in ("default.yaml", f"{env}.yaml"):
            fpath = config_dir / fname
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    merged.update(yaml.safe_load(f) or {})

        merged["env"] = env
        return cls(**merged)


settings = ServerSettings.load()