#!/bin/bash -l
# The -l flag makes this a login shell, which can help with environment setup.

#PBS -l select=1:ncpus=4:ngpus=1
#PBS -l place=vscatter:shared
#PBS -q gpu
#PBS -j oe
#PBS -N LLM_selection_rl

START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

# --- Robust Conda environment activation for non-interactive shells ---
echo "Sourcing Conda environment..."
source /home/luketou/miniconda3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "Error: Failed to source conda.sh. Please check the path."
    exit 1
fi
conda activate agent
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment 'agent'."
    exit 1
fi

# --- Set a writable cache directory for Hugging Face Transformers ---
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers
mkdir -p $TRANSFORMERS_CACHE

cd /home/luketou/GA_LLM_World_Model/llm_teach/

# Create log directory if it doesn't exist
mkdir -p log

echo "Installing/Verifying dependencies..."
# Reinstall faiss-cpu and numpy to ensure compatibility and fix AttributeError
conda install -c conda-forge faiss-cpu=1.7.4 numpy=1.23.5 sentence-transformers gxx_linux-64 -y

# LangGraph is typically safe with pip
pip install langgraph

echo "Starting LLM selection script..."
# Execute the new script
python llm_muvera_rl_critic.py --llm_option cerebras --tasks amlodipine --input_dir data/offspring --output_dir results/muvera_rl_critic_results --fg_dir data/functiongroup_offspring --model gpt-oss-120b --cerebras_api_keys "csk-yc5xd56kcxwc9x5y5rfc6mw95mfknd892mjjkhdyj39y898h" "csk-8f3ct6y23mw2fw3fdmyx8t4tx8rw8p85tdx9nrm8mv2t9m26" "csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2" "csk-22c9dktpnx6yc94rpv4h8xp952nvcnypnmn36nrk3ym953yf" "csk-n58d54hd8njxfnd225fkvhyj8wd28v2t656eexecxt3mh6t4" --max_generations 50 2>&1 | tee -a log/job_muvera_rl_critic.log


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate