#!/usr/bin/env bash
# =================== PBS directives ===================
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l place=pack:shared
#PBS -q gpu
#PBS -j oe
#PBS -N DPO_training

set -euo pipefail
START_TIME=$SECONDS

# ======== Basic config (edit as needed) ========
CSV_PATH=${CSV_PATH:-data/offspring/amlodipine.csv}
OUTDIR=${OUTDIR:-results/DPO_results}
WARMUP_GENS=${WARMUP_GENS:-30}
EPOCHS=${EPOCHS:-16}
DELTA_START=${DELTA_START:-0.05}
DELTA_END=${DELTA_END:-0.01}
PAIRS_PER_EPOCH=${PAIRS_PER_EPOCH:-12000}
BETA=${BETA:-0.12}
ONLINE_STEPS=${ONLINE_STEPS:-120}
REPLAY_K=${REPLAY_K:-20}
REF_REFRESH=${REF_REFRESH:-3}
PROXY_LEN_PENALTY=${PROXY_LEN_PENALTY:-0.0}
TOPM=${TOPM:-30}
TOPK=${TOPK:-10}
LR=${LR:-1e-4}

# ======== Env & project ========
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env

cd /home/luketou/GA_LLM_World_Model/llm_teach/
mkdir -p "${OUTDIR}"

# ======== GPUs from PBS (single-node) ========
if [[ -f "${PBS_GPUFILE:-}" ]]; then
  IDS=$(sed -E 's/.*gpu([0-9]+).*/\1/' "$PBS_GPUFILE" | paste -sd, -)
  export CUDA_VISIBLE_DEVICES=$(echo "${IDS}" | awk -F',' '{for(i=1;i<=NF;i++){if(!seen[$i]++){out=(out?out",":"")$i}} print out}')
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Visible GPU count
NPROC=$(python - <<'PY'
import os, torch
vis=os.environ.get("CUDA_VISIBLE_DEVICES","")
if vis:
    n=len([x for x in vis.split(",") if x.strip()!=""])
else:
    n=torch.cuda.device_count()
print(max(1,n))
PY
)
echo "nproc_per_node=${NPROC}"

# ======== Launch DDP training (always) ========
exec torchrun --standalone --nproc_per_node=${NPROC} \
  dpo_smiles_train.py \
    --csv "${CSV_PATH}" \
    --out "${OUTDIR}" \
    --warmup_gens "${WARMUP_GENS}" \
    --epochs "${EPOCHS}" \
    --delta_start "${DELTA_START}" \
    --delta_end "${DELTA_END}" \
    --pairs_per_epoch "${PAIRS_PER_EPOCH}" \
    --beta "${BETA}" \
    --pairs_mode hard --mix_alpha 0.7 \
    --online_steps "${ONLINE_STEPS}" \
    --replay_k "${REPLAY_K}" \
    --ref_refresh "${REF_REFRESH}" \
    --proxy_len_penalty "${PROXY_LEN_PENALTY}" \
    --topM "${TOPM}" \
    --topk "${TOPK}" \
    --lr "${LR}" \
    --cosine_schedule --warmup_ratio 0.05 \
    --amp \
    --layers 8 \
    --dist 2>&1 | tee log/dpo_training.log


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='
