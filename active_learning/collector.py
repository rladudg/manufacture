import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger
from config.settings import settings


def maybe_capture(image_bytes: bytes, result: dict) -> Optional[str]:
    if not settings.active_learning_enabled:
        return None

    detections = result.get("detections", [])
    top_conf = result.get("top_confidence", 0.0)

    bucket = None
    if len(detections) == 0:
        bucket = "no_detection"
    elif top_conf < settings.confidence_low_threshold:
        bucket = "low_conf"
    elif (len(detections) >= 2
          and detections[0]["confidence"] - detections[1]["confidence"] < 0.10):
        bucket = "ambiguous"

    if bucket is None:
        return None

    base = Path(settings.active_learning_dir) / bucket / datetime.now().strftime("%Y%m%d")
    base.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_path = base / f"{stamp}.jpg"
    meta_path = base / f"{stamp}.json"

    img_path.write_bytes(image_bytes)
    meta_path.write_text(json.dumps({
        "captured_at": datetime.now().isoformat(),
        "model_version": settings.model_version,
        "bucket": bucket,
        "result": result,
    }, ensure_ascii=False, indent=2))

    logger.info("Active learning capture: {bucket} {path}", bucket=bucket, path=img_path)
    return str(img_path)