#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg first: sudo apt-get install ffmpeg'; exit 1; }
"$PYTHON_BIN" -m venv .venv311
.venv311/bin/python -m pip install --upgrade pip
.venv311/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv311/bin/python -m pip install -r requirements.txt
if [ ! -d vjepa2/.git ]; then git clone https://github.com/facebookresearch/vjepa2.git vjepa2; fi
git -C vjepa2 checkout 204698b45b3712590f06245fbfba32d3be539812
if git -C vjepa2 apply --check ../patches/vjepa-official-url.patch 2>/dev/null; then
 git -C vjepa2 apply ../patches/vjepa-official-url.patch
else
 git -C vjepa2 apply --reverse --check ../patches/vjepa-official-url.patch
fi
.venv311/bin/python scripts/prepare_data.py
export TORCH_HOME="$PWD/cache"
.venv311/bin/python -c "import sys; sys.path.insert(0,'vjepa2'); from src.hub.backbones import vjepa2_1_vit_base_384; vjepa2_1_vit_base_384(pretrained=True); print('V-JEPA 2.1 checkpoint ready')"
echo 'Start with: CUDA_VISIBLE_DEVICES=0 bash prompt-studio/start.sh'
