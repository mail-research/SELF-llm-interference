from huggingface_hub import HfApi
from tqdm import tqdm

api = HfApi(token="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")

for step in tqdm([100, 150, 200, 250, 300, 350, 400, 450, 500], desc="Uploading..."):
    api.upload_folder(
        folder_path=f"checkpoints/rlvr-interfere/Qwen2.5-Math-1.5B-GRPO-filtered-max/global_step_{step}/actor_hf",
        repo_id="DatPySci/RLDI",
        path_in_repo=f"max_ppo/global_step_{step}",
        repo_type="model",
    )
