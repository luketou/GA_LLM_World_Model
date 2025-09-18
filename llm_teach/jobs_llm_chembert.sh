#!/bin/bash -l
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q gpu
#PBS -j oe
#PBS -N LLM_selection_rag

set -euo pipefail

START_TIME=$SECONDS

export CONDA_SOLVER=classic
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/vllm_env/lib:$HOME/miniconda3/lib:${LD_LIBRARY_PATH:-}"

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

conda activate vllm_env || { echo 'Failed to activate vllm_env'; exit 1; }

cd /home/luketou/GA_LLM_World_Model/llm_teach/

mkdir -p log

DEVICE=${DEVICE_OVERRIDE:-cuda}
if [[ "$DEVICE" == "cpu" ]]; then
    export PYTORCH_NO_CUDA=1
    export CUDA_VISIBLE_DEVICES=""
else
    unset PYTORCH_NO_CUDA
    # retain CUDA_VISIBLE_DEVICES assigned by scheduler
fi

python - <<'PY'
import os, torch
print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('torch.cuda.is_available=', torch.cuda.is_available())
print('torch.cuda.device_count=', torch.cuda.device_count())
PY

python llm_ChemBERTa_rag.py \
  --cerebras-api-keys \
    "csk-yc5xd56kcxwc9x5y5rfc6mw95mfknd892mjjkhdyj39y898h" \
    "csk-8f3ct6y23mw2fw3fdmyx8t4tx8rw8p85tdx9nrm8mv2t9m26" \
    "csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2" \
    "csk-22c9dktpnx6yc94rpv4h8xp952nvcnypnmn36nrk3ym953yf" \
    "csk-n58d54hd8njxfnd225fkvhyj8wd28v2t656eexecxt3mh6t4" \
    --device "$DEVICE" \
  --max-generation 100 2>&1 | tee -a log/job_llm_chemBERT.log

ELAPSED_TIME=$((SECONDS - START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $((ELAPSED_TIME/60)) 分鐘 $((ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate
