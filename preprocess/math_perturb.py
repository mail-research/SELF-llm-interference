import json
from datasets import load_dataset
import pandas as pd
import os
import argparse

def preprocess_fn(example):
    question = example.pop('problem')
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    question = f"{question}\n{instruction}"
    answer = example.pop('answer')
    split = example.pop("original_split")
    idx = example.pop("problem_id")

    data = {
        "data_source": "",
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer)
        },
        "extra_info": {
            'split': split,
            'index': idx
        }
    }
    return data

def train_preprocess_fn(example):
    question = example.pop('problem')
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    question = f"{question}\n{instruction}"
    answer = example.pop('answer')
    split = example.pop("original_split")
    idx = example.pop("problem_id")

    data = {
        "data_source": "",
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": str(answer)
        },
        "extra_info": {
            'split': split,
            'index': idx
        }
    }
    return data



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturb", type=str, default="simple")
    parser.add_argument("--output_folder", type=str, default="data/math_perturb")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    for perturb in ["simple", "hard"]:
        filepath = f"MATH-Perturb/math_perturb/math_perturb_{perturb}.jsonl"
        dataset = [json.loads(line) for line in open(filepath)]
        dataset = [preprocess_fn(line) for line in dataset]
        dataset = pd.DataFrame(dataset)
        dataset.to_parquet(os.path.join(args.output_folder, f"math_perturb_{perturb}.parquet"))
    

    train_dataset = pd.read_parquet(os.path.join(args.output_folder, "train.parquet"))
    prompts = train_dataset['question'].tolist()
    answers = train_dataset["answer"].tolist()
    extra_infos = train_dataset['extra_info'].tolist()
    idxs = [ele["index"] for ele in extra_infos]
    
    all_examples = []
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    for prompt, answer, idx in zip(prompts, answers, idxs):
        prompt = f"{prompt}\n{instruction}"
        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": prompt
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(answer)
            },
            "extra_info": {
                'split': "train",
                'index': idx
            }
        }
        all_examples.append(data)
    all_examples = pd.DataFrame(all_examples)
    all_examples.to_parquet(os.path.join(args.output_folder, f"math_train.parquet"))