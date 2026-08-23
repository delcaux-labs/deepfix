"""S3 model weights utilities for DeepFix SDK."""

import io
import os
import pathlib
from typing import Any, Optional

import boto3


def push_model_to_s3(
    model: Any,
    s3_bucket: str,
    model_name: str = "model",
    s3_prefix: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region_name: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Push a model artifact (PyTorch, Scikit-learn, joblib/pickle, or file) to S3 and return the S3 URI."""
    if isinstance(model, str) and model.startswith("s3://"):
        return model

    prefix = s3_prefix.strip("/") if s3_prefix else f"models/{model_name}"

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    s3_client = session.client(
        "s3", endpoint_url=endpoint_url or os.getenv("AWS_ENDPOINT_URL")
    )

    # 1. If model is a file path / pathlib.Path on disk
    if isinstance(model, (str, pathlib.Path)) and os.path.exists(str(model)):
        path = pathlib.Path(model)
        if path.is_file():
            filename = path.name
            s3_key = f"{prefix}/{filename}" if prefix else filename
            s3_client.upload_file(str(path), s3_bucket, s3_key)
            return f"s3://{s3_bucket}/{s3_key}"

    # 2. If model is a PyTorch Module or state dict
    try:
        import torch

        if isinstance(model, torch.nn.Module):
            s3_key = (
                f"{prefix}/{model_name}.pt" if prefix else f"{model_name}.pt"
            )
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, s3_bucket, s3_key)
            return f"s3://{s3_bucket}/{s3_key}"
        elif isinstance(model, dict) and any(
            isinstance(v, torch.Tensor) for v in model.values()
        ):
            s3_key = (
                f"{prefix}/{model_name}.pt" if prefix else f"{model_name}.pt"
            )
            buffer = io.BytesIO()
            torch.save(model, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, s3_bucket, s3_key)
            return f"s3://{s3_bucket}/{s3_key}"
    except (ImportError, Exception):
        pass

    # 3. If model is a Scikit-Learn or other Python model object, serialize using joblib or pickle
    buffer = io.BytesIO()
    try:
        import joblib

        joblib.dump(model, buffer)
        ext = "joblib"
    except Exception:
        import pickle

        pickle.dump(model, buffer)
        ext = "pkl"

    s3_key = (
        f"{prefix}/{model_name}.{ext}" if prefix else f"{model_name}.{ext}"
    )
    buffer.seek(0)
    s3_client.upload_fileobj(buffer, s3_bucket, s3_key)
    return f"s3://{s3_bucket}/{s3_key}"
