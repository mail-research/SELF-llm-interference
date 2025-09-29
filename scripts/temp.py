from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from typing import Union, List
# from grader import math_equal
import itertools


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



if __name__=="__main__":
    output_path = "results/Qwen2.5-Math-1.5B-GRPO/minerva_step-0_pass144.parquet"
    dataset = pd.read_parquet(output_path)
    data_sources = dataset["data_source"]
    reward_model_data = dataset['reward_model']
    responses = dataset['responses']

    total = []
    correct = []

    for i in tqdm(range(len(dataset))):
        response_lst = responses[i]
        data_source = data_sources[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            score = rllm_reward_fn_math(data_source, r, ground_truth)
            score_lst.append(score)
        
        total.append(len(score_lst))
        correct_reward = [1 if score == 1.0 else 0 for score in score_lst]
        correct_indices = [idx for idx, score in enumerate(score_lst) if score == 1.0]
        correct.append(sum(correct_reward))
    
    # json_path = os.path.join(output_dir, 'pass.jsonl')
    # dataset_name = os.path.basename(config.data.path)
    # row_data = {
    #     'model_path': config.model.path,
    #     'dataset': dataset_name,
    #     'raw_scores': correct,
    #     'total': total[0],
    # }
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    total = np.array(total)
    correct = np.array(correct)
    print(correct)
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
                for k in ks if (total >= k).all()}
    print(pass_at_k)
    # for k, v in pass_at_k.items():
    #     row_data[k] = v
    # 'seed': seed
    # for k, v in pass_at_k.items():
    #     row_data[k] = v

    # Check if file exists