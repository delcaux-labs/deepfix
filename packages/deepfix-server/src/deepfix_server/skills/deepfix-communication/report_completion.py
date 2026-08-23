import argparse
import json
import os
import sys
import urllib.request
from typing import Any, Dict


def parse_metrics(metrics_arg: Any) -> Dict[str, Any]:
    if not metrics_arg:
        return {}
    if isinstance(metrics_arg, dict):
        return metrics_arg
    try:
        return json.loads(metrics_arg)
    except Exception:
        parsed = {}
        for pair in str(metrics_arg).split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    parsed[k.strip()] = float(v.strip())
                except ValueError:
                    parsed[k.strip()] = v.strip()
        return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report completion to DeepFix Server webhook."
    )
    parser.add_argument(
        "--webhook-url",
        default=None,
        help="Webhook URL to POST to (defaults to DEEPFIX_WEBHOOK_URL env)",
    )
    parser.add_argument(
        "--job-id",
        default=None,
        help="The Job ID assigned to this fix task (defaults to DEEPFIX_JOB_ID env)",
    )
    parser.add_argument(
        "--success",
        action="store_true",
        default=None,
        help="Whether the fix session was successful",
    )
    parser.add_argument(
        "--failed",
        action="store_true",
        default=False,
        help="Flag indicating the fix session failed",
    )
    parser.add_argument(
        "--status",
        choices=["COMPLETED", "FAILED", "completed", "failed"],
        default=None,
        help="Job status string (COMPLETED or FAILED)",
    )
    parser.add_argument(
        "--s3-weights-uri",
        default=None,
        help="S3 URI where the fixed model weights are saved",
    )
    parser.add_argument(
        "--final-run-id",
        "--run-id",
        dest="final_run_id",
        default=None,
        help="Final MLflow Run ID",
    )
    parser.add_argument(
        "--applied-fixes",
        "--fixes",
        dest="fixes",
        nargs="*",
        default=[],
        help="List of fixes applied during the session",
    )
    parser.add_argument(
        "--final-metrics",
        default="{}",
        help="JSON string or key=val pairs of final evaluation metrics",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Markdown or text summary of the fix results",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Total iterations executed",
    )

    args = parser.parse_args()

    job_id = args.job_id or os.getenv("DEEPFIX_JOB_ID")
    if not job_id:
        print(
            "Error: --job-id or DEEPFIX_JOB_ID environment variable is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    webhook_url = (
        args.webhook_url
        or os.getenv("DEEPFIX_WEBHOOK_URL")
        or "http://host.docker.internal:4141/webhook/completion"
    )

    if args.failed:
        is_success = False
    elif args.status is not None:
        is_success = args.status.upper() == "COMPLETED"
    elif args.success is not None:
        is_success = bool(args.success)
    else:
        is_success = True

    status_str = "COMPLETED" if is_success else "FAILED"
    final_metrics = parse_metrics(args.final_metrics)

    payload = {
        "job_id": job_id,
        "success": is_success,
        "status": status_str,
        "final_metrics": final_metrics,
        "applied_fixes": args.fixes,
        "run_id": args.final_run_id,
        "s3_weights_uri": args.s3_weights_uri,
        "summary": args.summary,
        "iteration": args.iteration,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"Submitting completion report to webhook: {webhook_url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            print(
                f"Webhook notified successfully: HTTP {response.status} - "
                f"{response_body}"
            )
    except Exception as e:
        print(f"Failed to notify webhook: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
