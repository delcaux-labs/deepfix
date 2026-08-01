#!/bin/bash

# Set environment variables to handle Unicode characters
export PYTHONIOENCODING=utf-8

echo "============================================================"
echo "          DeepFix SDK - E2E Workflow Tests"
echo "============================================================"

# Check if test.env exists
if [ ! -f "test.env" ]; then
    echo "[ERROR] test.env file not found in the root directory."
    echo "Please create it with DEEPFIX_API_KEY and DEEPFIX_TEST_API_URL."
    exit 1
fi

# Run Tabular E2E Test
echo
echo "[1/3] Running Tabular Workflow E2E Test..."
uv run --env-file test.env pytest tests/test_tabular_workflow_e2e.py -s
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "[FAIL] Tabular E2E Test failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "[PASS] Tabular Workflow successful."

# Run NLP E2E Test
echo
echo "[2/3] Running NLP Workflow E2E Test..."
uv run --env-file test.env pytest tests/test_nlp_workflow_e2e.py -s
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "[FAIL] NLP E2E Test failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "[PASS] NLP Workflow successful."

# Run Vision E2E Tests (MNIST and FoodWaste)
echo
echo "[3/3] Running Vision Workflow E2E Tests..."
uv run --env-file test.env pytest tests/test_vision_workflow_e2e.py -s
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo
    echo "[FAIL] Vision E2E Tests failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "[PASS] Vision Workflow successful."

echo
echo "============================================================"
echo "          ALL E2E WORKFLOW TESTS PASSED"
echo "============================================================"
