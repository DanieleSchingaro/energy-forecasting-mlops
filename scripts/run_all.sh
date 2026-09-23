#scripts/run_all.sh

#!/usr/bin/env bash
# Esegue l'intera pipeline e scrive un log datato in logs/.
#
#   ./scripts/run_all.sh              pipeline completa
#   ./scripts/run_all.sh train        solo lo stage train e cio' che ne dipende
#   ./scripts/run_all.sh --force      riesegue tutto ignorando la cache DVC
#
# Gli argomenti vengono passati a `dvc repro`. Funziona da Git Bash su Windows
# e da una shell Linux: l'ambiente virtuale viene individuato in entrambi i casi.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ -d ".venv/Scripts" ]; then
    BIN=".venv/Scripts"   # Windows
    EXT=".exe"
elif [ -d ".venv/bin" ]; then
    BIN=".venv/bin"       # Linux / macOS
    EXT=""
else
    echo "Ambiente virtuale non trovato in .venv" >&2
    exit 1
fi

PY="$BIN/python$EXT"
DVC="$BIN/dvc$EXT"

# DVC lancia i comandi degli stage in una sotto-shell: senza questo, il `python`
# degli stage sarebbe quello di sistema e non troverebbe il pacchetto energy.
export PATH="$PWD/$BIN:$PATH"

for tool in "$PY" "$DVC"; do
    if [ ! -x "$tool" ]; then
        echo "$tool non trovato: installa le dipendenze nell'ambiente virtuale" >&2
        exit 1
    fi
done

mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"

pipeline() {
    echo "=== $(date '+%F %T') pipeline avviata"
    "$PY" --version
    echo

    echo "--- test"
    "$PY" -m pytest -q
    echo

    echo "--- dvc repro"
    "$DVC" repro "$@"
    echo

    echo "--- metriche"
    "$DVC" metrics show
    echo "=== $(date '+%F %T') pipeline completata"
}

pipeline "$@" 2>&1 | tee "$LOG"
echo "Log salvato in $LOG"