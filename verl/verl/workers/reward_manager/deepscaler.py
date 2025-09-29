import torch
from verl import DataProto
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math

class DeepscalerRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            score = rllm_reward_fn_math(data_source, llm_solution=sequences_str, ground_truth=ground_truth)
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=48) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
            }
        else:
            return reward_tensor

class MixDeepscalerRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        ref_reward_tensor = torch.zeros_like(data.batch['ref_responses'], dtype=torch.float32)

        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any
        #import threading
        # Thread-safe dict for tracking printed data sources
        # print_lock = threading.Lock()
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            ref_response_ids = data_item.batch['ref_responses'] 
            ref_valid_response_length = data_item.batch['ref_attention_mask'][prompt_length:].sum()
            ref_valid_response_ids = ref_response_ids[:ref_valid_response_length]
            ref_sequences = torch.cat((valid_prompt_ids, ref_valid_response_ids))
            ref_sequences_str = self.tokenizer.decode(ref_sequences)
            data_source = data_item.non_tensor_batch['data_source']
            ref_score = rllm_reward_fn_math(data_source, llm_solution=ref_sequences_str, ground_truth=ground_truth)
            ref_score = 1.0 if ref_score == 1.0 else 0.0

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            # select rm_score
            score = rllm_reward_fn_math(data_source, llm_solution=sequences_str, ground_truth=ground_truth)
            # with print_lock:
            #     if data_source not in already_print_data_sources:
            #         already_print_data_sources[data_source] = 0

            #     if already_print_data_sources[data_source] < self.num_examine:
            #         already_print_data_sources[data_source] += 1
            #         print(sequences_str)      
            return i, score, valid_response_length, ref_score, ref_valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=48) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length, ref_score, ref_valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score
            ref_reward_tensor[i, ref_valid_response_length - 1] = ref_score
            
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {"ref_reward_tensor": ref_reward_tensor}
            }
        else:
            return reward_tensor