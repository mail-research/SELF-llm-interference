from torch import nn, optim
from collections import defaultdict
from transformers import AutoTokenizer
import os
import json
import pandas as pd
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
from typing import Union, List
import itertools
import numpy as np

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["aime", "amc", "math", "minerva", "olympiad_bench"])
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--is_llama", type=int, default=0, choices=[0,1]) 
    parser.add_argument("--num_slices", type=int, default=2)
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()

    N_RESPONSES=256
    data_name = args.dataset
    step = args.step
    model_path = args.model_path.split("/")[-1]
    num_slices = args.num_slices
    is_llama = bool(args.is_llama)

    folder_path = "data/deepscaler"
    if is_llama:
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        df = pd.read_parquet(os.path.join(folder_path, f"{data_name}_simple.parquet"))
        all_prompts = df['prompt'].tolist()
        all_prompts = [ele[0]["content"] for ele in all_prompts]
    else:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        df = pd.read_parquet(os.path.join(folder_path, f"{data_name}.parquet"))
        all_prompts = df['prompt'].tolist()
        all_prompts = [chat.tolist() for chat in all_prompts]
        all_prompts = tokenizer.apply_chat_template(all_prompts, add_generation_prompt=True, tokenize=False)
    print(len(all_prompts))
    print(all_prompts[0])

    ground_truths = df['reward_model'].tolist()
    ground_truths = [ele['ground_truth'] for ele in ground_truths]
    from openai import OpenAI
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    models = client.models.list()
    model = models.data[0].id

    # completion = client.completions.create(
    #     model=model,
    #     prompt=all_prompts,
    #     n=N_RESPONSES,
    #     temperature=0.6,
    #     top_p=0.95,
    #     max_tokens=16384,
    #     stream=False
    # )
    # solution_strs = [ele.text for ele in completion.choices]
    # solution_strs = [solution_strs[i * N_RESPONSES:(i + 1) * N_RESPONSES] for i in range(len(solution_strs) // N_RESPONSES)]

    sub_responses = N_RESPONSES // num_slices
    solution_strs = defaultdict(list)

    for i in range(num_slices):
        if is_llama:
            completion = client.completions.create(
                model=model,
                prompt=all_prompts,
                n=sub_responses,
                temperature=0.6,
                top_p=0.95,
                stop=["Question", "Answer"],
                timeout=None,
                max_tokens=16384,
                stream=False
            )
        else:
            completion = client.completions.create(
                model=model,
                prompt=all_prompts,
                n=sub_responses,
                temperature=0.6,
                top_p=0.95,
                max_tokens=16384,
                timeout=None,
                stream=False
            )
        sub_solution_strs = [ele.text for ele in completion.choices]
        sub_solution_strs = [sub_solution_strs[i * sub_responses:(i + 1) * sub_responses] for i in range(len(sub_solution_strs) // sub_responses)]
        # B x M
        for i in range(len(sub_solution_strs)):
            solution_strs[i].extend(sub_solution_strs[i])

    total = []
    correct = []

    for i in range(len(df)):
        response_lst = solution_strs[i]
        ground_truth = ground_truths[i]
        score_lst = []
        for r in response_lst:
            score = rllm_reward_fn_math("deepscaler", r, ground_truth)
            score_lst.append(score)
        total.append(len(score_lst))
        correct_reward = [1 if score == 1.0 else 0 for score in score_lst]
        correct.append(sum(correct_reward))

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    results = {
        "model_path": f"SimpleRL-{model_path}-step-{step}",
        "correct": correct,
        "total": total[0],
        "dataset": data_name,
    }
    total = np.array(total)
    correct = np.array(correct)
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
                for k in ks if (total >= k).all()}
    print(pass_at_k)
    for k, v in pass_at_k.items():
        results[k] = v

    with open("results/results.jsonl", "a") as f:
        json.dump(results, f)
        f.write("\n")
            