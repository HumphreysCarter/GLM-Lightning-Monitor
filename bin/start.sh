#!/usr/bin/env bash
set -euo pipefail

API_CMD=("uvicorn" "src.api_server:app" "--host" "0.0.0.0" "--port" "${API_PORT}" "--workers" "${UVICORN_WORKERS}")
WORKER_CMD=("python" "-m" "src.glm_ingest")

echo "[entry] starting API: ${API_CMD[*]}"
"${API_CMD[@]}" &
API_PID=$!

echo "[entry] starting ingest worker: ${WORKER_CMD[*]}"
"${WORKER_CMD[@]}" &
WORKER_PID=$!

term() {
  echo "[entry] received signal, stopping..."
  kill -TERM "$API_PID" "$WORKER_PID" 2>/dev/null || true
}
trap term INT TERM

# wait for either to exit, then stop the other
set +e
wait -n "$API_PID" "$WORKER_PID"
EXIT_CODE=$?
echo "[entry] one process exited (code $EXIT_CODE); terminating the other..."
kill -TERM "$API_PID" "$WORKER_PID" 2>/dev/null || true
wait || true
exit $EXIT_CODE
