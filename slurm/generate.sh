#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval-qwen2.5-7b-4k.out
#SBATCH -e log/error-eval-qwen2.5-7b-4k.out
#SBATCH --mem=256G
#SBATCH --time=1-00:00:0q

#SBATCH --job-name=eval-qwen7b
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:4

set -x
nvidia-smi
source ~/.shadow1
conda activate llm_churn
OUTPUT_DIR="data/deepscaler"  # Add default output directory
MAX_LENGTH=8192 # Default max response length
TP_SIZE=2  # Default tensor parallel size

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}" 
echo "Number of Passes: ${N_PASSES}"
echo "Max Response Length: ${MAX_LENGTH}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# MODEL_PATH=checkpoints/deepscaler/deepscaler-Qwen2.5-3B/global_step_500/actor_hf
# MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
STEPS=("50" "100" "150" "200" "250" "300" "350" "400" "450" "500")
MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# model_name
# MODEL_PATH=Qwen/Qwen2.5-Math-7B
# model_name=Qwen2.5-Math-7B-GRPO
# MODEL_PATH=checkpoints/logic-kk/logic-kk-Qwen-3B/global_step_600/actor_hf
# MODEL_PATH=Qwen/Qwen2.5-7B
# Loop through all datatypes

# for step in "${STEPS[@]}"; do
# MODEL_PATH=checkpoints/rlvr-interfere/${model_name}/global_step_${step}/actor_hf

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=data/deepscaler/deepscaler_train.parquet \
    data.n_samples=4 \
    data.output_path=${OUTPUT_DIR}/deepscaler_Llama-3.2-3B-Instruct.parquet \
    data.batch_size=4096\
    model.path=${MODEL_PATH} \
    rollout.temperature=0.9 \
    rollout.response_length=${MAX_LENGTH} \
    rollout.prompt_length=1024 \
    rollout.top_k=-1 \
    rollout.top_p=1.0 \
    rollout.gpu_memory_utilization=0.8\
    rollout.tensor_model_parallel_size=${TP_SIZE}

# done