#!/bin/bash
#SBATCH -c 16 # request two cores 
#SBATCH -p kisski-h100
#SBATCH -o log/grpo-llama3-8b.out
#SBATCH -e log/error-grpo-llama3-8b.out
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=grpo-llama-7b
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4
set -x
# 
nvidia-smi
source ~/.shadow1
conda activate llm_churn

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
# Parse command line arguments
cl
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    # MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
    MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"
    # MODEL_PATH="Qwen/Qwen2.5-1.5B"
    # MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
fi

    # trainer.experiment_name='deepscaler-Llama-3.2-3B-Instruct' \
# Train over a single node, 8 A100-80GB GPUs.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=data/deepscaler/Qwen2.5-Math-1.5B_filtered.parquet\
    data.val_files=data/deepscaler/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=rllm \
    trainer.experiment_name='Qwen2.5-Math-1.5B-GRPO-filtered' \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rlvr-interfere' \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=2 "${@:1}"