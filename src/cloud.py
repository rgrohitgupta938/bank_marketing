# src/cloud.py

import os
import boto3


def s3_upload_file(local_path: str, bucket: str, key: str) -> None:
    """
    Upload a single file to S3.
    """
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)


def upload_dir_to_s3(local_dir: str, bucket: str, prefix: str) -> None:
    """
    Upload an entire directory recursively to S3, preserving relative paths.
    """
    prefix = prefix.strip("/")

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = f"{prefix}/{rel_path}"
            s3_upload_file(local_path, bucket, key)
