#!/bin/bash

#SBATCH --job-name=llm_teach_cerebras    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --cpus-per-task=4       ## 該 task 索取 32 CPUs
#SBATCH --gres=gpu:1             ## 每個節點索取 8 GPUs
#SBATCH --account GOV113073
#SBATCH --partition=qtest      ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)
#SBATCH -o suz_cerebras.out                                                     
#SBATCH -e suz_cerebras.err 

START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

module purge
module load cuda/12.8
module load miniconda3
conda activate GA_agent

cd /home/luketou123/GA_LLM_World_Model/llm_teach
# source GA_agent/bin/activate

# Create log directory if it doesn't exist
mkdir -p log

echo "Starting LLM selection for multiple tasks..."

python llm_select.py --tasks median2 scaffold_hop isomer_c9h10n2o2pf2cl isomer_c11h24 valsartan_smarts \
    --max_model_len 8000 --llm_option 'cerebras' --max_candidates 100 2>&1 | tee -a log/llm_selection2.log

# another api key
python llm_select.py --model qwen-3-235b-a22b --tasks median2 scaffold_hop isomer_c9h10n2o2pf2cl isomer_c11h24 valsartan_smarts --cerebras_api_key csk-r45fve4re56cxcmmvj8j4f5c5hrvt893deyk9p9pwre36t38 --max_model_len 8000 --llm_option 'cerebras' --max_candidates 100 2>&1 | tee -a log/llm_selection_api.log

# python llm_select.py --task "amlodipine" --max_model_len 4380 --output_dir 'results_cerebras' --cerebras_api_key 'csk-38kjr3mnp22x9w2m2wejdej5m55cpkwhmh3p6p6f3wt2wxw2' --llm_option 'cerebras' --model 'llama-3.3-70b'

echo "LLM selection completed!"


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate