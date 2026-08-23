import argparse
import json
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser(description="Report completion to DeepFix Server webhook.")
    parser.add_argument("--webhook-url", required=True, help="Webhook URL to POST to")
    parser.add_argument("--job-id", required=True, help="The Job ID assigned to this fix task")
    parser.add_argument("--success", action="store_true", help="Whether the fix session was successful")
    parser.add_argument("--final-run-id", required=True, help="Final MLflow Run ID")
    parser.add_argument("--fixes", nargs="*", default=[], help="List of fixes applied")
    parser.add_argument("--final-metrics", type=json.loads, default={}, help="JSON string of final metrics")

    args = parser.parse_args()

    payload = {
        "job_id": args.job_id,
        "success": args.success,
        "final_metrics": args.final_metrics,
        "applied_fixes": args.fixes,
        "run_id": args.final_run_id,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        args.webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            print(f"Webhook notified successfully: {response.status}")
    except Exception as e:
        print(f"Failed to notify webhook: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
