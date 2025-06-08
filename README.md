# GA LLM World Model

This project implements a simplified molecular design workflow combining a LangGraph based language model agent, a Graph GA generator, and a world model with a risk gate.

## Installation

```bash
conda create -n molwm python=3.9 && conda activate molwm
pip install torch==2.* torch_geometric==2.* rdkit-pypi \
    guacamol==0.5.2 guacamol-baselines langgraph==0.4.*
```

## Usage

Pretrain the world model:

```bash
python training/train_pretrain.py --epochs 50
```

Fine-tune the model:

```bash
python training/train_finetune.py --epochs 30
```

Single SMILES inference:

```bash
python inference.py \
   --smiles "CC1=CC(=O)NC(=O)N1" \
   --task_name "logP" --op ">=" --value 2.5
```

Run the GuacaMol benchmark suite:

```bash
python benchmark/guacamol_runner.py --oracle_budget 200
```
