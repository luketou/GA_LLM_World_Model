#!/bin/bash

#SBATCH --job-name=re_train    ## job name
#SBATCH --nodes=1                ## 索取 1 節點
#SBATCH --cpus-per-task=32       ## 該 task 索取 32 CPUs
#SBATCH --gres=gpu:8             ## 每個節點索取 8 GPUs
#SBATCH --account MST108470
#SBATCH --partition=gtest       ## gtest 為測試用 queue，後續測試完可改 gp1d(最長跑1天)、gp2d(最長跑2天)、p4d(最長跑4天)

module purge
module load miniconda3
module load cuda/11.7
conda activate retrieve

MODEL_DIR="output/suzuki_1003"
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

  for split in test train
  do
    echo $split

    python -m tevatron.faiss_retriever \
      --query_reps ${MODEL_DIR}/${split}.pkl \
      --passage_reps ${MODEL_DIR}/corpus.pkl \
      --depth 20 \
      --batch_size -1 \
      --save_json \
      --save_ranking_to ${MODEL_DIR}/${split}_rank.json
  done
