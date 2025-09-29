# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import json
import os
import csv
import hydra
import numpy as np
import ray
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
# from tabulate import tabulate
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from typing import Union, List
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

@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init()
        # ray.init(
        #     num_cpus=config.ray_init.num_cpus,
        # )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    if os.path.exists(config.data.output_path):
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        dataset = pd.read_parquet(config.data.output_path)
    else:
        local_path = copy_to_local(config.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # seed = config.rollout.seed
        
        if config.rollout.temperature == 0.0:
            assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
        assert config.data.n_samples >= 1, "n_samples should always >= 1"

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        dataset = pd.read_parquet(config.data.path)
        chat_lst = dataset[config.data.prompt_key].tolist()

        chat_lst = [chat.tolist() for chat in chat_lst]
        print("-"*100)
        print(chat_lst[0][0]["content"])
        print("-"*100)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        config_batch_size = config.data.batch_size
        num_batch = -(-total_samples // config_batch_size)
        output_lst = []

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            repeated_chat_lst = []
            for chat in batch_chat_lst:
                repeated_chat_lst.extend([chat] * config.data.n_samples)
            
            # if tokenizer.chat_template is not None:
            inputs = tokenizer.apply_chat_template(
                repeated_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            )
            # else:
            #     inputs = tokenizer([chat[0]["content"] for chat in repeated_chat_lst], max_length=config.rollout.prompt_length, return_tensors="pt", padding=True, truncation=True)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            position_ids = compute_position_id_with_mask(attention_mask)
            batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

            data = DataProto.from_dict(batch_dict)
            # data.meta_info = {"seed": seed}
            data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)

            # START TO GENERATE FOR n_samples TIMES
            print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
            output_padded = wg.generate_sequences(data_padded)
            output = unpad_dataproto(output_padded, pad_size=pad_size)
            output_texts = []
            for i in range(len(output)):
                data_item = output[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = data_item.batch["responses"][:valid_response_length]
                response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                output_texts.append(response_str)

            output_lst.extend(output_texts)

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        total_samples = len(output_lst)
        n_data = total_samples // config.data.n_samples
        output_lst = np.array(output_lst, dtype=object).reshape(n_data, config.data.n_samples).tolist()
        dataset["responses"] = output_lst
        dataset.to_parquet(config.data.output_path)
        
        # add to the data frame
        data_sources = dataset["data_source"]
        reward_model_data = dataset['reward_model']
        responses = dataset['responses']

        total = []
        correct = []

        for i in range(len(dataset)):
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
            correct.append(sum(correct_reward))
        
        json_path = os.path.join(output_dir, 'pass.jsonl')
        dataset_name = os.path.basename(config.data.path)
        row_data = {
            'model_path': config.model.path,
            'dataset': dataset_name,
            'raw_scores': correct,
            'total': total[0],
        }
        ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        total = np.array(total)
        correct = np.array(correct)
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
                    for k in ks if (total >= k).all()}
        print(pass_at_k)
        for k, v in pass_at_k.items():
            row_data[k] = v
        # 'seed': seed
        # for k, v in pass_at_k.items():
        #     row_data[k] = v

        # Check if file exists
        # file_exists = os.path.isfile(json_path)
        print("JSON path:", json_path)
        # Write to CSV
        with open(json_path, 'a+') as f:
            json.dump(row_data, f)
            f.write('\n')

if __name__ == "__main__":
    main()
