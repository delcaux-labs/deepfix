import argparse
import os
import sys


def download_dataset(
    run_id: str, artifact_path: str = "dataset", output_dir: str = "./data"
) -> str:
    """Download dataset artifact from an MLflow run to the specified directory."""
    import mlflow

    os.makedirs(output_dir, exist_ok=True)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path, dst_path=output_dir
    )
    print(f"Dataset downloaded successfully to: {local_path}")
    return local_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download dataset artifacts from MLflow."
    )
    parser.add_argument("--run-id", required=True, help="MLflow Run ID")
    parser.add_argument(
        "--artifact-path", default="dataset", help="Artifact path in MLflow"
    )
    parser.add_argument(
        "--output-dir", default="./data", help="Output directory on disk"
    )

    args = parser.parse_args()
    try:
        download_dataset(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"Failed to download dataset from MLflow: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
