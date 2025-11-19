#!/bin/bash
set -e

echo "==> Python being used: $(which python)"
echo "==> Checking Hypercorn in current env"
python -m pip show hypercorn || echo "!! Hypercorn NOT installed"

python -m hypercorn editor_api:app --bind 0.0.0.0:${PORT:-8000}