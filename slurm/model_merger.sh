# STEPS=("50" "100" "150" "200" "250")
STEPS=("50" "100" "200" "250" "300" "350" "400" "450" "500")
STEPS=("500")
# "300" "
# STEPS=("350" "400" "450" "500")
# STEPS=("50" "100" "150" "200" "250" "300")
# STEPS=("150" "200" "250" "300" "350")
# STEPS=("450" "500")

# model_path=Qwen2.5-Math-1.5B-GRPO-filtered-max
# model_path=Qwen2.5-Math-1.5B-GRPO-one-shot
model_path=Qwen2.5-Math-7B-reg

data=rlvr-interfere
base_model_path=Qwen/Qwen2.5-Math-7B
# base_model_path=Qwen/Qwen2.5-Math-7B
# base_model_path=meta-llama/Llama-3.2-3B-Instruct

for step in "${STEPS[@]}"; do
python verl/scripts/model_merger.py merge\
    --backend fsdp \
    --hf_model_path  $base_model_path\
    --local_dir checkpoints/${data}/${model_path}/global_step_${step}/actor\
    --target_dir checkpoints/${data}/${model_path}/global_step_${step}/actor_hf \
    --tie-word-embedding
done