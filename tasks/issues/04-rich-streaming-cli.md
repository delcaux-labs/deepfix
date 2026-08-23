# 04 — Rich Streaming CLI with Live Progress & Metrics Tables

**What to build:** A developer-friendly streaming CLI interface using Rich formatting that provides real-time visibility into the autonomous fix process. As the agent iterates inside the sandbox, the CLI displays a dynamic progress bar for iterations, a live phase indicator (e.g. `Diagnosing`, `Synthesizing Fix`, `Training Run #2`, `Evaluating`, `Uploading Weights to S3`), and formatted tables showing metric improvements (baseline vs. intermediate runs) without requiring a browser UI.

**Blocked by:** 01 — Minimal End-to-End Fix Job Tracer Bullet

**Status:** ready-for-agent

- [ ] CLI displays a Rich progress bar tracking current iteration versus `--max-iterations`.
- [ ] Real-time status stream prints agent activity log events and phase transitions with formatted timestamps and spinners/icons.
- [ ] Summary table prints intermediate MLflow metric results (iteration, loss, accuracy, F1, ROC-AUC) on stdout.
- [ ] Gracefully handles Ctrl+C / SIGINT by prompting the user to either cancel the server job or detach cleanly while the job continues in background.
- [ ] CLI flags `--target-metric`, `--target-value`, `--max-iterations`, and `--poll-interval` configure client behavior properly.
- [ ] Typecheck and lint pass across `deepfix-sdk`.
