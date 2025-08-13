#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_PROJECT=MoELoRA

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

CONFIGS_FOLDER="configs/model_configs"
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 accelerate launch --config_file ${CONFIGS_FOLDER}/multi_gpu.yaml --num_processes 2 -m src.run   ${CONFIGS_FOLDER}/moelora_task_aware.yaml
