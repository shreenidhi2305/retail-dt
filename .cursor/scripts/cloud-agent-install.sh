#!/usr/bin/env bash
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"

pip install --user -r requirements.txt

if [[ ! -f models/demand_model.pkl ]]; then
  python3 train_model.py
fi
