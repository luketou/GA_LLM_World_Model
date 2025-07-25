#!/bin/bash

#SBATCH --job-name=2llm_teach2    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --cpus-per-task=32       ## 該 task 索取 32 CPUs
#SBATCH --gres=gpu:8             ## 每個節點索取 8 GPUs
#SBATCH --account GOV113073
#SBATCH --partition=gp1d         ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)
#SBATCH -o llm_select2.out                                                     
#SBATCH -e llm_select2.err 

START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

module purge
module load cuda/12.8
module load miniconda3
conda activate GA_agent

cd /home/luketou123/GA_LLM_World_Model/llm_teach
# source GA_agent/bin/activate

TASKS=(
    osimertinib
    fexofenadine
    ranolazine
    amlodipine
    perindopril
    sitagliptin
    zaleplon
    celecoxib
    troglitazone
    thiothixene
    median1
    median2
    scaffold_hop
    isomer_c9h10n2o2pf2cl
    isomer_c11h24
    valsartan_smarts
)


# Create log directory if it doesn't exist
mkdir -p log

wallet

python llm_select.py --tasks scaffold_hop isomer_c9h10n2o2pf2cl isomer_c11h24 valsartan_smarts \
    --max_model_len 4380 --llm_option 'vllm' --max_candidates 100 2>&1 | tee -a log/llm_selection2.log

ELAPSED_TIME=$(($SECONDS - $START_TIME))

wallet

echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='

conda deactivate