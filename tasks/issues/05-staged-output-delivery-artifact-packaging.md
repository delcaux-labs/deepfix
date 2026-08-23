# 05 — Staged Output Delivery & Artifact Packaging

**What to build:** The final stage of the autonomous fix CLI workflow where repaired model assets and diagnostic reports are cleanly packaged and exported. Once the server marks a job complete, the CLI creates a staged output folder at `./deepfix_output/<job_id>/` containing the repaired standalone `train_fixed.py` script, a comprehensive `summary_report.md` (highlighting initial defects, remediations applied, metric deltas, MLflow run URLs, and S3 weight URIs), `metrics.json`, and local model weights, ensuring the user's working directory is never directly overwritten.

**Blocked by:** 03 — Diagnostic-to-Prompt Synthesis & Autonomous Agent Execution Engine, 04 — Rich Streaming CLI with Live Progress & Metrics Tables

**Status:** completed

- [x] CLI generates staged output directory `./deepfix_output/<job_id>/` (or user-specified `--output-dir`).
- [x] Staged directory contains:
  - `train_fixed.py`: Clean, standalone, runnable Python training script incorporating the winning fixes.
  - `summary_report.md`: Formatted Markdown report detailing original diagnostic issues, applied code changes, baseline vs. final metrics, and S3 weights link (`s3://...`).
  - `metrics.json`: JSON file with structured metrics before and after the fix.
  - `model_artifacts/`: Downloaded model checkpoint/weights from S3 or MLflow.
- [x] CLI outputs a prominent summary banner with relative paths to all generated artifacts and exits with code `0` on success or non-zero on failure.
- [x] Typecheck and lint pass across modified packages.
