# src/s3_artifacts.py

import json
from pathlib import Path
import boto3
import joblib

CACHE_DIR = Path(".cache_s3")
CACHE_DIR.mkdir(exist_ok=True)


def _s3():
    return boto3.client("s3")


def read_latest_run_id(bucket: str, prefix: str) -> str:
    """
    Reads s3://{bucket}/{prefix}/latest_run.json which must contain {"run_id": "..."}.
    """
    key = f"{prefix.strip('/')}/latest_run.json"
    obj = _s3().get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    return data["run_id"]


def download_if_missing(bucket: str, key: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        _s3().download_file(bucket, key, str(local_path))
    return local_path


def load_preprocessor(bucket: str, prefix: str, run_id: str):
    base = prefix.strip("/")
    s3_key = f"{base}/models/{run_id}/preprocessor.pkl"
    local_path = CACHE_DIR / "models" / run_id / "preprocessor.pkl"
    download_if_missing(bucket, s3_key, local_path)
    return joblib.load(local_path)


def load_model(bucket: str, prefix: str, run_id: str, model_filename: str):
    base = prefix.strip("/")
    s3_key = f"{base}/models/{run_id}/{model_filename}"
    local_path = CACHE_DIR / "models" / run_id / model_filename
    download_if_missing(bucket, s3_key, local_path)
    return joblib.load(local_path)


def clear_cache():
    if CACHE_DIR.exists():
        for p in sorted(CACHE_DIR.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            else:
                try:
                    p.rmdir()
                except OSError:
                    pass
