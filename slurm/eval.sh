#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval-qwen2.5-7b-deepscaler-math.out
#SBATCH -e log/error-eval-qwen2.5-7b-deepscaler-math.out
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=eval-qwen7b-deepscaler-math
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

source ~/.shadow1
conda activate llm_churn
set -x

# Default values
# MODEL_PATH=Qwen/Qwen2.5-1.5B
# MODEL_PATH=meta-llama/Llama-3.2-3B-Instruct
# DATATYPES=("aime" "aime25" "math" "minerva")
# "math" "minerva")
DATATYPES=("aime" "aime25") 
OUTPUT_DIR="results"  # Add default output directory
# N_PASSESS=(144 112 256 1024) # Add default number of passes
# N_PASSESS=(112 144 256 1024) # Add default number of passes
# N_PASSESS=(112 144) # Add default number of passes
N_PASSESS=(512) # Add default number of passes

MAX_LENGTH=3072  # Default max response length
TP_SIZE=2  # Default tensor parallel size

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}" 
echo "Number of Passes: ${N_PASSES}"
echo "Max Response Length: ${MAX_LENGTH}"
echo "Tensor Parallel Size: ${TP_SIZE}"

# STEPS=("50" "100" "150" "200" "250" "300" "350" "400" "450")
STEPS=("500")
# )"250" "300" "350" "400" "450" "500")
# STEPS=("400" "450" "500")

# STEPS=("150" "200" "250" "300" "350" "400" "450" "500")
# STEPS=('500')

# model_name=Qwen2.5-Math-7B-GRPO-filtered-max
model_name=Qwen2.5-Math-7B-reg
# model_name=Qwen2.5-Math-1.5B-buffer
# model_name=Qwen2.5-Math-7B-GRPO
# MODEL_PATH=Qwen/Qwen2.5-Math-7B
# step=0

for N_PASSES in "${N_PASSESS[@]}"; do
for DATA_TYPE in "${DATATYPES[@]}"; do
for step in "${STEPS[@]}"; do

MODEL_PATH=checkpoints/rlvr-interfere/${model_name}/global_step_${step}/actor_hf
# MODEL_PATH=checkpoints/rlvr-interference/deepscaler-Llama-3.2-3B-Instruct/global_step_${step}/actor_hf

python3 -m scripts.eval \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=data/deepscaler/${DATA_TYPE}.parquet \
    data.n_samples=${N_PASSES} \
    data.output_path=${OUTPUT_DIR}/${model_name}/${DATA_TYPE}_step-${step}_pass${N_PASSES}.parquet \
    data.batch_size=512 \
    model.path=${MODEL_PATH} \
    rollout.temperature=0.6 \
    rollout.response_length=${MAX_LENGTH} \
    rollout.prompt_length=1024 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.gpu_memory_utilization=0.75 \
    rollout.tensor_model_parallel_size=${TP_SIZE}

done
done
done
