from datasets import load_dataset
import pandas as pd
from typing import Dict, Optional, Any

def process_fn(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    question = example.pop('problem')
    instruction = "Let's think step by step and output the final answer within \\boxed{}."
    idx = example.pop('id')
    question = f"{question}\n{instruction}"
    answer = example.pop('answer')
    data = {
        "data_source": "",
        "prompt": [{
            "role": "user",
            "content": question
        }],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer
        },
        "extra_info": {
            'split': "test",
            'index': idx
        }
    }
    return data

if __name__=="__main__":
    train_dataset = load_dataset("math-ai/aime25", split='test')
    train_dataset = train_dataset.map(process_fn, num_proc=8, batched=False)
    test_data = []
    for sample in train_dataset:
        test_data.append(sample)

    test_df = pd.DataFrame(test_data)
    print(test_df)
    test_df.to_parquet("data/deepscaler/aime25.parquet")