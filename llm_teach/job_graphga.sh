#!/usr/bin/bash

#PBS -l select=1:ncpus=8:ngpus=1
#PBS -l place=vscatter:shared
#PBS -q gpu
#PBS -j oe
#PBS -N GA

# 記錄開始時間
START_TIME=$SECONDS

# 清除環境變數避免衝突
unset LD_PRELOAD
unset BNB_CUDA_VERSION 
unset LD_LIBRARY_PATH

# 設置正確的 CUDA 相關環境變數
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 設置 Conda 環境
source /home/luketou/miniconda3/bin/activate agent

# 確保使用正確的系統庫版本（避免 GLIBCXX 不兼容問題）
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 加載必要的 BLAS 庫
export LD_PRELOAD=$CONDA_PREFIX/lib/libopenblas.so:$LD_PRELOAD

# 設定工作目錄
cd /home/luketou/GA_LLM_World_Model/llm_teach/

echo "=========================================================="
echo "Starting on : $(date)"
echo "Running on node : $(hostname)"
echo "Current directory : $(pwd)"
echo "Current job ID : $PBS_JOBID"
echo "GPU Device: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

rm -rf log/main.log
mkdir -p /log
touch log/main.log

# ...existing code...

TASKS=(
# osimertinib
# fexofenadine
# ranolazine
amlodipine
# perindopril
# sitagliptin
# zaleplon
# cobimetinib
# cns_mpo
# scaffold_hop
# decoration_hop
# weird_physchem
# valsartan_smarts
# median1
# median2
# isomer_c11h24
# isomer_c9h10n2o2pf2cl
# celecoxib
# troglitazone
# thiothixene
# mestranol
)

for TASK in \"${TASKS[@]}\"
do
    echo \"===== Running task: $TASK =====\" 2>&1 | tee -a log/main.log
    python -m graph_ga.goal_directed_generation --population_size 15 --offspring_size 30 --task $TASK --smiles_file "/home/luketou/LLM_AI_agent/Agent_predictor/data/guacamol_v1_all.txt" 2>&1 | tee -a log/main.log
done


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "任務完成，耗時 $(($ELAPSED_TIME/60)) 分鐘 $(($ELAPSED_TIME%60)) 秒"
echo "Job Ended at $(date)"
echo '=========================================================='


