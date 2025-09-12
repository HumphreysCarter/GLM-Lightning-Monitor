#!/usr/bin/env bash
set -euo pipefail

API_CMD=("uvicorn" "src.api_server:app" "--host" "0.0.0.0" "--port" "${API_PORT}" "--workers" "${UVICORN_WORKERS}")
INGEST_CMD=("python" "-m" "src.glm_ingest")
NOTIFY_CMD=("python" "-m" "src.glm_notify")

echo "[entry] starting API server: ${API_CMD[*]}"
"${API_CMD[@]}" &
API_PID=$!

echo "[entry] starting GLM ingest worker: ${INGEST_CMD[*]}"
"${INGEST_CMD[@]}" &
WORKER_PID=$!

echo "[entry] starting GLM notify worker: ${NOTIFY_CMD[*]}"
"${NOTIFY_CMD[@]}" &
NOTIFY_PID=$!

term() {
  echo "[entry] received signal, stopping..."
  kill -TERM "$API_PID" "$WORKER_PID" "$NOTIFY_PID" 2>/dev/null || true
}
trap term INT TERM

# wait for any to exit, then stop the others
set +e
wait -n "$API_PID" "$WORKER_PID" "$NOTIFY_PID"
EXIT_CODE=$?
echo "[entry] one process exited (code $EXIT_CODE); terminating the others..."
kill -TERM "$API_PID" "$WORKER_PID" "$NOTIFY_PID" 2>/dev/null || true
wait || true
exit $EXIT_CODE
