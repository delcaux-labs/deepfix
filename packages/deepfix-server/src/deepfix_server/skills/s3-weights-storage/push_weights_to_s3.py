import argparse
import json
import os
import pathlib
import sys
from typing import Optional

import boto3


def push_weights_to_s3(
    weights_path: str,
    s3_bucket: Optional[str] = None,
    job_id: Optional[str] = None,
    s3_prefix: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    region_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> str:
    """Upload model weights / checkpoint file to S3 and return the S3 URI."""
    bucket = (
        s3_bucket
        or os.getenv("AWS_S3_BUCKET")
        or os.getenv("DEEPFIX_S3_BUCKET")
    )
    if not bucket:
        raise ValueError(
            "S3 bucket must be specified via --s3-bucket or "
            "AWS_S3_BUCKET / DEEPFIX_S3_BUCKET env vars."
        )

    job = job_id or os.getenv("DEEPFIX_JOB_ID", "default_job")
    endpoint = endpoint_url or os.getenv("AWS_ENDPOINT_URL")
    region = (
        region_name
        or os.getenv("AWS_DEFAULT_REGION")
        or os.getenv("AWS_REGION", "us-east-1")
    )
    access_key = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

    path = pathlib.Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights file does not exist at: {weights_path}")

    filename = path.name
    if s3_prefix:
        s3_key = f"{s3_prefix.strip('/')}/{filename}"
    else:
        s3_key = f"{job}/weights/{filename}"

    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
    )
    s3_client = session.client("s3", endpoint_url=endpoint)

    print(f"Uploading {weights_path} to s3://{bucket}/{s3_key}...")
    s3_client.upload_file(str(path), bucket, s3_key)

    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"Successfully uploaded model weights: {s3_uri}")
    return s3_uri


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload model weights / checkpoint directly to S3."
    )
    parser.add_argument(
        "--weights-path",
        required=True,
        help="Path to local weights file or checkpoint on disk",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Target S3 bucket name (defaults to AWS_S3_BUCKET env)",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="Job ID assigned to this fix task (defaults to DEEPFIX_JOB_ID env)",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Optional S3 prefix override (defaults to <job_id>/weights/)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="S3 endpoint URL (defaults to AWS_ENDPOINT_URL env)",
    )
    parser.add_argument(
        "--region-name",
        default=None,
        help="AWS region (defaults to AWS_DEFAULT_REGION env)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON payload",
    )

    args = parser.parse_args()

    try:
        s3_uri = push_weights_to_s3(
            weights_path=args.weights_path,
            s3_bucket=args.s3_bucket,
            job_id=args.job_id,
            s3_prefix=args.s3_prefix,
            endpoint_url=args.endpoint_url,
            region_name=args.region_name,
        )
        if args.json:
            print(json.dumps({"s3_weights_uri": s3_uri, "status": "success"}))
    except Exception as exc:
        print(f"Error uploading weights to S3: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
