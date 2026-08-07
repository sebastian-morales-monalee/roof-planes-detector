#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-gpu}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ "${DEVICE}" != "gpu" && "${DEVICE}" != "cpu" ]]; then
  echo "Usage: ./setup_env.sh [gpu|cpu]" >&2
  exit 2
fi

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

CURRENT_TORCH_VERSION="$(
  "${VENV_DIR}/bin/python" -c \
    'import torch; print(torch.__version__)' 2>/dev/null || true
)"

if [[ "${DEVICE}" == "gpu" ]]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu130"
  EXPECTED_SUFFIX="+cu130"
else
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  EXPECTED_SUFFIX="+cpu"
fi

TORCH_INSTALL_ARGS=()
if [[ "${CURRENT_TORCH_VERSION}" != *"${EXPECTED_SUFFIX}" ]]; then
  TORCH_INSTALL_ARGS+=(--force-reinstall)
fi

"${VENV_DIR}/bin/python" -m pip install \
  "${TORCH_INSTALL_ARGS[@]}" \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url "${TORCH_INDEX}"

"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PY
