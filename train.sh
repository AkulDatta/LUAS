#!/bin/bash

# Hugging Face Token
HF_TOKEN="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"

# Export the token as an environment variable
export HUB_TOKEN=$HF_TOKEN
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

NUM_GPUS=1

get_gpu_count() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l
    else
        echo 0
    fi
}

TOTAL_GPUS=$(get_gpu_count)
USE_GPUS=$NUM_GPUS

if [ $TOTAL_GPUS -eq 0 ]; then
    echo "Error: No GPUs found in the system"
    exit 1
fi

if [ $NUM_GPUS -gt $TOTAL_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $TOTAL_GPUS available. Using $TOTAL_GPUS GPUs."
    USE_GPUS=$TOTAL_GPUS
fi

CUDA_DEVICES=""
for ((i=0; i<$USE_GPUS; i++)); do
    if [ $i -eq 0 ]; then
        CUDA_DEVICES="$i"
    else
        CUDA_DEVICES="${CUDA_DEVICES},$i"
    fi
done

DATASET_NAME="agent_sft_act_dataset"

export PYTHONPATH=$(pwd)
echo "${PYTHONPATH}"

cd ./training_scripts

WANDB_PROJECT=""

MODEL_TYPE="7b"
MODEL_NAME="meta-llama/Llama-2-${MODEL_TYPE}-hf"
# MODEL_TYPE="8b"
# MODEL_NAME="meta-llama/Llama-3.1-${MODEL_TYPE}"

DATASET_DIR="../generation/multiwoz/converters/woz.2.2.gen"

LR=2e-5
BATCH_SIZE=4
EPOCH=1

TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR##*/}"

# First training phase
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --nnodes 1 \
    --nproc_per_node $USE_GPUS \
    --master_port=1201 \
    ./llama_finetuning.py \
    --enable_fsdp \
    --model_name "${MODEL_NAME}" \
    --dataset "${DATASET_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --save_model \
    --pure_bf16 \
    --output_dir "./${DATASET_NAME}.${TAG}" \
    --lr "${LR}" \
    --valid_batch_size "${BATCH_SIZE}" \
    --train_batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${BATCH_SIZE}" \
    --num_epochs "${EPOCH}" \
    --evaluation_steps 200 \
    --check_point_steps 1000000 \
    --wandb_name "${TAG}" \
    --wandb_project "${WANDB_PROJECT}"

python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path "./${DATASET_NAME}.${TAG}/epoch_000" \
    --consolidated_model_path "./${DATASET_NAME}.${TAG}/epoch_000.hf" \
    --HF_model_path_or_name "${MODEL_NAME}"

PRE_TRAIN_MODEL="./${DATASET_NAME}.${TAG}/epoch_000.hf"

# Training on real data
DATASET_DIR="../generation/multiwoz/converters/woz.2.2.real"
TAG="${MODEL_TYPE}.${LR}.full.B${BATCH_SIZE}.E${EPOCH}.${DATASET_DIR##*/}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun \
    --nnodes 1 \
    --nproc_per_node $USE_GPUS \
    --master_port=1201 \
    ./llama_finetuning.py \
    --enable_fsdp \
    --model_name "${PRE_TRAIN_MODEL}" \
    --dataset "${DATASET_NAME}" \
    --dataset_dir "${DATASET_DIR}" \
    --save_model \
    --pure_bf16 \
    --output_dir "./${DATASET_NAME}.${TAG}.Real" \
    --lr "${LR}" \
    --valid_batch_size "${BATCH_SIZE}" \
    --train_batch_size "${BATCH_SIZE}" \
    --micro_batch_size "${BATCH_SIZE}" \
    --num_epochs "${EPOCH}" \
    --evaluation_steps 200 \
    --check_point_steps 1000000 \
    --wandb_name "${TAG}" \
    --wandb_project "${WANDB_PROJECT}"

python inference/checkpoint_converter_fsdp_hf.py \
    --fsdp_checkpoint_path "./${DATASET_NAME}.${TAG}.Real/epoch_000" \
    --consolidated_model_path "./${DATASET_NAME}.${TAG}.Real/epoch_000.hf" \
    --HF_model_path_or_name "${MODEL_NAME}"