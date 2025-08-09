#!/usr/bin/bash

#PBS -l select=1:ncpus=32:ngpus=4
#PBS -l place=vscatter:shared
#PBS -q gpu
#PBS -j oe
#PBS -N LLM_selection_rl

START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

# --- Robust Conda environment activation for non-interactive shells ---
echo "Sourcing Conda environment..."
source /home/luketou/miniconda3/etc/profile.d/conda.sh

# First unset any problematic LD_LIBRARY_PATH
unset LD_LIBRARY_PATH

# Activate the environment
conda activate vllm_env

# --- Set a writable cache directory for Hugging Face Transformers ---
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
mkdir -p $TRANSFORMERS_CACHE

cd /home/luketou/GA_LLM_World_Model/llm_teach/

# Create log directory if it doesn't exist
mkdir -p log

# Use the conda environment's Python explicitly
PYTHON_EXEC=$(which python)
echo "Using Python: $PYTHON_EXEC"

# Set LD_LIBRARY_PATH to use conda's libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Starting LLM selection script..."
# Execute the new script with explicit python path
python llm_muvera_rl_critic.py --llm_option vllm --hf_model openai/gpt-oss-20b --tasks amlodipine --input_dir data/offspring --output_dir results/muvera_rl_critic_results --fg_dir data/functiongroup_offspring --model gpt-oss-20b --cerebras_api_keys "csk-yc5xd56kcxwc9x5y5rfc6mw95mfknd892mjjkhdyj39y898h" "csk-8f3ct6y23mw2fw3fdmyx8t4tx8rw8p85tdx9nrm8mv2t9m26" "csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2" "csk-22c9dktpnx6yc94rpv4h8xp952nvcnypnmn36nrk3ym953yf" "csk-n58d54hd8njxfnd225fkvhyj8wd28v2t656eexecxt3mh6t4" --max_generations 50 2>&1 | tee -a log/job_muvera_rl_critic_gpu.log


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate
