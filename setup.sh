#!/usr/bin/env bash
# setup.sh — create a local virtualenv and install MLX.
#
# MLX is Apple-Silicon only (it runs on the Mac GPU via Metal). We pin to Python 3.13
# because MLX does not yet publish wheels for Python 3.14.
# Override by setting PY=python3.x before running, e.g.:  PY=python3.12 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PY:-python3.13}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: $PY not found. Install it (brew install python@3.13) or set PY=..." >&2
  exit 1
fi

echo "creating venv with $($PY --version) ..."
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "done. next steps:"
echo "  source .venv/bin/activate"
echo "  python data.py     # download + tokenize Tiny Shakespeare"
echo "  python train.py    # train the model"
echo '  python sample.py --prompt "ROMEO:"'
