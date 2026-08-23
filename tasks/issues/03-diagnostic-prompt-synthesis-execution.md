# 03 — Diagnostic-to-Prompt Synthesis & Autonomous Agent Execution Engine

**What to build:** The core autonomous repair loop connecting the diagnostic system to the OpenHands agent execution engine. When a fix request is processed, the server extracts the detailed failure modes synthesized by `DiagnosticSystem.arun` (such as severe feature multicollinearity, class imbalance, missing EDA, and small sample size risk) and compiles them into a structured prompt for the OpenHands agent. The server then boots the agent in the sandbox to autonomously create `train.py`, run iterative training experiments logged to MLflow, evaluate against target metrics, and upload winning weights to S3.

**Blocked by:** 02 — OpenHands Sandbox Skills with S3 Weights Push & Webhook Callback

**Status:** completed

- [x] `OpenHandsExecutor` compiles diagnostic findings and modeling remediation guidelines into a system prompt for the agent.
- [x] Prompt explicitly instructs the agent to address specific failure modes:
  - Multicollinearity: apply regularization (L1/L2), feature reduction on high-PPS features, or tree-based models.
  - Class Imbalance: implement class weighting / `scale_pos_weight` and prioritize recall/F1/ROC-AUC over pure accuracy.
  - Validation: use stratified k-fold cross-validation instead of single train/test split.
  - Dataset-Only Mode: when no model is provided, focus on dataset partitioning repair (leak-free stratified splitting, class balance preservation, and feature filtering).
  - S3 Upload: push best model weights or partitioned datasets using `s3-weights-storage`.
- [x] Server initiates background agent execution using `DockerWorkspace` upon `POST /v2/fix`.
- [x] Agent iterates, executes `train.py` or partitioning scripts, logs metrics to MLflow, and terminates cleanly via the communication skill webhook.
- [x] Server endpoint `POST /v2/fix/{job_id}/cancel` and SDK `cancel_fix_job` support cancelling running background fix sessions.
- [x] Typecheck and lint pass.
