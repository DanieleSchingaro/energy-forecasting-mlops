#scripts/mlflow_ui.sh

#!/usr/bin/env bash
# Avvia la UI di MLflow sullo stesso store usato dagli script (sqlite:///mlflow.db).
# Resta in primo piano: interrompila con Ctrl+C.
#
#   ./scripts/mlflow_ui.sh            porta 5000
#   ./scripts/mlflow_ui.sh 5050       porta a scelta

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -d ".venv/Scripts" ]; then
    BIN=".venv/Scripts"
    EXT=".exe"
else
    BIN=".venv/bin"
    EXT=""
fi

PORT="${1:-5000}"

echo "MLflow UI su http://127.0.0.1:$PORT"
exec "$BIN/mlflow$EXT" ui --backend-store-uri sqlite:///mlflow.db --port "$PORT"