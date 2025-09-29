import pandas as pd
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from pandas import DataFrame
from datasets import load_dataset, concatenate_datasets

if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    # df = pd.read_parquet("data/deepscaler/deepscaler_Qwen2.5-Math-1.5B.parquet")
    df = pd.read_parquet("data/deepscaler/deepscaler_Qwen2.5-Math-1.5B_prob.parquet")
    
    responses = df["responses"].tolist()
    prompts = df["prompt"].tolist()
    extra_infos = df["extra_info"].tolist()
    reward_models = df["reward_model"].tolist()
    data_sources = df["data_source"].tolist()
    
    all_prompts = []
    all_responses = []
    all_extra_infos = []
    all_reward_models = []
    all_ds_srcs = []

    for prompt, resp_lst, extra_info, reward_model, ds_src, in zip(prompts, responses, extra_infos, reward_models, data_sources):
        # chat = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        for r in resp_lst:
            all_prompts.append(prompt)
            all_responses.append(r)
            all_extra_infos.append(extra_info)
            all_reward_models.append(reward_model)
            all_ds_srcs.append(ds_src)
        
    new_df = pd.DataFrame()
    print(len(all_prompts))
    print(len(all_responses))
    print(len(extra_infos))
    new_df["prompt"] = all_prompts
    new_df["response"] = all_responses
    new_df['extra_info'] = all_extra_infos
    new_df['reward_model'] = all_reward_models
    new_df['data_source'] = all_ds_srcs
    
    new_df = shuffle(new_df, random_state=42).reset_index()
    print(new_df)
    new_df.to_parquet("data/deepscaler/deepscaler_Qwen2.5-Math-1.5B_prob_flatten.parquet")