# =================== PBS directives ===================
#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l place=pack:shared
#PBS -q gpu
#PBS -j oe
#PBS -N DPO_training

# =================== Config (edit as needed) ===================
# MODE: "single" for one training run; "optuna" for hyperparameter tuning
MODE=${MODE:-single}

CSV_PATH=${CSV_PATH:-data/offspring/amlodipine.csv}
OUTDIR=${OUTDIR:-results/DPO_results}
STUDY_NAME=${STUDY_NAME:-dpo_hpo_sqlite}
OPTUNA_DB=${OPTUNA_DB:-sqlite:///optuna.db}
TRIALS=${TRIALS:-40}            # total trials for optuna mode
TRIALS_PER_WORKER=${TRIALS_PER_WORKER:-0}  # if 0, auto = ceil(TRIALS/NUM_GPUS)

# Base hyperparameters (both modes will reuse these unless Optuna overrides)
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
COSINE_SCHED=${COSINE_SCHED:-1}   # 1: enable, 0: disable

# =================== Cluster/Conda setup ===================
set -euxo pipefail
START_TIME=$SECONDS

# Work around conda-libmamba-solver GLIBCXX issues on cluster
export CONDA_SOLVER=classic
export CONDA_NO_PLUGINS=1
# Make sure user-installed GCC libstdc++ doesn't override Conda's
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's#:/home/luketou/gcc-10.2/lib64##g' -e 's#/home/luketou/gcc-10.2/lib64:##g' -e 's#/home/luketou/gcc-10.2/lib64##g')
fi

echo "Sourcing Conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm_env
conda config --show solver || true
python -V

# Reduce tokenizer parallelism noise
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Safer NCCL defaults for single-node
export NCCL_DEBUG=warn
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^lo,docker0

cd /home/luketou/GA_LLM_World_Model/llm_teach/
mkdir -p log "${OUTDIR}"

# Map PBS allocated GPUs to CUDA_VISIBLE_DEVICES if provided
if [[ -f "${PBS_GPUFILE:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=$(awk -F'-gpu' '{print $2}' "$PBS_GPUFILE" | paste -sd, -)
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# Count visible GPUs
VISIBLE_GPU_COUNT=$(python - <<'PY'
import os, torch
vis=os.environ.get("CUDA_VISIBLE_DEVICES","")
if vis:
    n=len([x for x in vis.split(",") if x.strip()!=""])
else:
    n=torch.cuda.device_count()
print(max(1,n))
PY
)
echo "VISIBLE_GPU_COUNT=${VISIBLE_GPU_COUNT}"

# Helper: assemble common args
COMMON_ARGS=(
  --csv "${CSV_PATH}"
  --out "${OUTDIR}"
  --warmup_gens "${WARMUP_GENS}"
  --epochs "${EPOCHS}"
  --delta_start "${DELTA_START}"
  --delta_end "${DELTA_END}"
  --pairs_per_epoch "${PAIRS_PER_EPOCH}"
  --beta "${BETA}"
  --pairs_mode hard --mix_alpha 0.7
  --online_steps "${ONLINE_STEPS}"
  --replay_k "${REPLAY_K}"
  --ref_refresh "${REF_REFRESH}"
  --proxy_len_penalty "${PROXY_LEN_PENALTY}"
  --topM "${TOPM}"
  --topk "${TOPK}"
  --lr "${LR}"
)
if [[ "${COSINE_SCHED}" == "1" ]]; then
  COMMON_ARGS+=( --cosine_schedule --warmup_ratio 0.05 )
fi
COMMON_ARGS+=( --amp )

# =================== Modes ===================

if [[ "${MODE}" == "single" ]]; then
  echo "==> MODE=single (DDP via torchrun if multi-GPU)"
  DIST_FLAG=""
  if [[ ${VISIBLE_GPU_COUNT} -gt 1 ]]; then
    DIST_FLAG="--dist"
  fi
  echo "Starting single training run ..."
  torchrun --standalone --nproc_per_node=${VISIBLE_GPU_COUNT} \
    dpo_smiles_train.py \
      "${COMMON_ARGS[@]}" \
      ${DIST_FLAG} \
    2>&1 | tee -a log/dpo_single.log

elif [[ "${MODE}" == "optuna" ]]; then
  echo "==> MODE=optuna (Hyperparameter tuning with Optuna)"
  echo "Study Name: ${STUDY_NAME}"
  echo "Storage: ${OPTUNA_DB}"
  # Write combined log
  LOGFILE="log/dpo_optuna_$(date +%Y%m%d_%H%M%S).log"

  if [[ ${VISIBLE_GPU_COUNT} -eq 1 ]]; then
    echo "Running Optuna on a single GPU (sequential trials) ..." | tee -a "${LOGFILE}"
    python dpo_smiles_train.py \
      "${COMMON_ARGS[@]}" \
      --optuna --trials "${TRIALS}" \
      --study_name "${STUDY_NAME}" \
      --storage "${OPTUNA_DB}" \
      --direction maximize \
      --nosave \
      2>&1 | tee -a "${LOGFILE}"
  else
    echo "Running Optuna with multi-GPU parallel workers (one worker per GPU) ..." | tee -a "${LOGFILE}"
    # Determine trials per worker (ceil division) if not set
    if [[ "${TRIALS_PER_WORKER}" -le 0 ]]; then
      TRIALS_PER_WORKER=$(( (TRIALS + VISIBLE_GPU_COUNT - 1) / VISIBLE_GPU_COUNT ))
    fi
    echo "Total trials: ${TRIALS}; Workers=${VISIBLE_GPU_COUNT}; Trials/worker=${TRIALS_PER_WORKER}" | tee -a "${LOGFILE}"

    # Split CUDA_VISIBLE_DEVICES into an array
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
      # fallback to numeric 0..N-1
      GPU_IDS=($(seq 0 $((VISIBLE_GPU_COUNT-1))))
    fi

    # Launch one worker per GPU, each with isolated OUTDIR and distinct optuna seed
    pids=()
    idx=0
    for gid in "${GPU_IDS[@]}"; do
      WORK_OUT="${OUTDIR}/optuna_worker_${gid}"
      mkdir -p "${WORK_OUT}"
      (
        export CUDA_VISIBLE_DEVICES="${gid}"
        echo "[Worker ${gid}] starting ..." | tee -a "${LOGFILE}"
        python dpo_smiles_train.py \
          "${COMMON_ARGS[@]/${OUTDIR}/${WORK_OUT}}" \
          --optuna --trials "${TRIALS_PER_WORKER}" \
          --study_name "${STUDY_NAME}" \
          --storage "${OPTUNA_DB}" \
          --direction maximize \
          --optuna_seed $((42 + idx)) \
          --nosave \
          2>&1 | tee -a "${LOGFILE}"
        echo "[Worker ${gid}] done." | tee -a "${LOGFILE}"
      ) &
      pids+=($!)
      idx=$((idx+1))
    done

    # Wait all workers
    for pid in "${pids[@]}"; do
      wait "$pid"
    done
    echo "All Optuna workers completed." | tee -a "${LOGFILE}"
  fi

else
  echo "Unknown MODE='${MODE}'. Use MODE=single or MODE=optuna." >&2
  exit 1
fi

# =================== Footer ===================
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate