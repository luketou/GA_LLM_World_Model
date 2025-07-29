#!/bin/bash

#PBS -l select=1:ncpus=1:ngpus=1
# PBS -l place=vscatter:shared
#PBS -q gpu
#PBS -j oe
#PBS -N LLM_selection_api

START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

# module purge
# module load cuda/12.8
# module load miniconda3
conda init
conda activate agent

cd /home/luketou/GA_LLM_World_Model/llm_teach/
# source GA_agent/bin/activate

# Create log directory if it doesn't exist
mkdir -p log

echo "Starting LLM selection for multiple tasks..."

# median2 scaffold_hop isomer_c9h10n2o2pf2cl isomer_c11h24 valsartan_smarts
python llm_select_with_fg.py --tasks amlodipine --input_dir data/offspring --output_dir results/results_qwen_3_235b_a22b \
    --cerebras_api_keys "csk-yc5xd56kcxwc9x5y5rfc6mw95mfknd892mjjkhdyj39y898h" "csk-8f3ct6y23mw2fw3fdmyx8t4tx8rw8p85tdx9nrm8mv2t9m26" "csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2" "csk-22c9dktpnx6yc94rpv4h8xp952nvcnypnmn36nrk3ym953yf" "csk-n58d54hd8njxfnd225fkvhyj8wd28v2t656eexecxt3mh6t4" \
    --model qwen-3-235b-a22b --max_model_len 20000 --max_token 10000 --llm_option 'cerebras' --max_candidates 100 2>&1 | tee -a log/job_cerebras.log


# another api key
# python llm_select.py --model qwen-3-235b-a22b --tasks median2 scaffold_hop isomer_c9h10n2o2pf2cl isomer_c11h24 valsartan_smarts --cerebras_api_key csk-r45fve4re56cxcmmvj8j4f5c5hrvt893deyk9p9pwre36t38 --max_model_len 8000 --llm_option 'cerebras' --max_candidates 100 2>&1 | tee -a log/llm_selection_api.log

# python llm_select.py --task "amlodipine" --max_model_len 4380 --output_dir 'results_cerebras' --cerebras_api_key 'csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2' --llm_option 'cerebras' --model 'llama-3.3-70b'

echo "LLM selection completed!"


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate