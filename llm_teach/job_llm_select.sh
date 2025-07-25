#!/bin/bash
#SBATCH --job-name=1llm_teach
#SBATCH --nodes=2               ## 索取 1 節點
#SBATCH --ntasks=2                # 2 tasks: head and worker
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=32      # 每個 task 分配 32 CPUs
#SBATCH --gres=gpu:8            # 每個節點索取 8 GPUs
#SBATCH --exclusive
#SBATCH --account=GOV113073
#SBATCH --partition=gp1d        ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)
#SBATCH -o llm_select1.out
#SBATCH -e llm_select1.err


START_TIME=$SECONDS

export HF_HOME=/work/luketou123/llm_model

module purge
module load cuda/12.8
module load miniconda3
conda activate GA_agent

cd /home/luketou123/GA_LLM_World_Model/llm_teach
mkdir -p log

# Determine head node and IP
head_node=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
head_ip=$(srun --nodes=1 --ntasks=1 -w $head_node hostname -I | awk '{print $1}')

echo "SLURM_PROCID = $SLURM_PROCID"
echo "Head node    = $head_node ($head_ip)"
echo "This host    = $(hostname) ($(hostname -I | awk '{print $1}'))"

# Start Ray head or worker based on PROCID
if [ "$SLURM_PROCID" -eq 0 ]; then
  echo ">>> Starting Ray head"
  ray start --head \
    --node-ip-address=$head_ip \
    --port=6379 \
    --num-cpus=$SLURM_CPUS_ON_NODE \
    --num-gpus=8
else
  echo ">>> Starting Ray worker"
  my_ip=$(hostname -I | awk '{print $1}')
  ray start --address=${head_ip}:6379 \
    --node-ip-address=$my_ip \
    --num-cpus=$SLURM_CPUS_ON_NODE \
    --num-gpus=8
fi

# Wait for cluster to form
sleep 30
ray status

# Only head node submits the vLLM job
if [ "$SLURM_PROCID" -eq 0 ]; then
  echo ">>> Running vLLM task"
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    python llm_select.py \
      --tasks fexofenadine \
      --llm_option vllm \
      --max_model_len 8000 \
      --max_candidates 100 \
      --tensor_parallel_size 8 \
      --pipeline_parallel_size 2 \
      --distributed_executor_backend ray
fi

# Stop Ray and cleanup (only on head process)
if [ "$SLURM_PROCID" -eq 0 ]; then
  ray stop
  ELAPSED_TIME=$((SECONDS - START_TIME))
  echo "=========================================================="
  echo "Task completed in $((ELAPSED_TIME/60)) min $((ELAPSED_TIME%60)) sec"
  echo "Job ended at $(date)"
  echo "=========================================================="
  conda deactivate
fi