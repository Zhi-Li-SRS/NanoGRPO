import dataclasses
import gc
import math 
from collections import defaultdict
from typing import Callable, List
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_types import Episode, MiniBatch
from qwen2_llm import Qwen2Config, Transformer
from tokenizer import Tokenizer

@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    """
    Implement inference and generate the texts and calculate the reward

    Args:
        model (Transformer): the pretrained model to generate the texts
        batch (MiniBatch): the batch of data including questions and context
        tokenizer (Tokenizer): the tokenizer to transform the texts to tokens or reverse
        max_gen_len (int): the maximum length of the generated texts
        num_answer_per_question (int): the number of answers to generate for each question
        reward_function (Callable): the function to calculate the reward
        device (torch.device): the device to run the model
        dtype (torch.dtype): the data type to run the model

    Returns:
        List[Episode]: the list of episodes including the generated texts and related information
    """
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.ens_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Get the prefix token ids and the size of batch
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question # bsz = batch_size * num_answer_per_question
    
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len 
    
    # Initialize the kv cache
    model.init_kv_cache(bsz, total_len, device, dtype)
    
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    
    # Fill the prefix token ids into the tokens tensor
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    
    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        
        print(
            f"\r* Generating trajectories {cur_pos - min_prompt_len:>4d}/{total_len - min_prompt_len:>4d}", 
            flush=True,
            end="",
        )
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            # Get the logits of the current position
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos) # (bsz, seq_len（1）, vocab_size)
        
        probs = torch.softmax(logits[:, -1], dim=-1) # (bsz, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1) # (bsz, 1)
        next_token = next_token.reshape(-1) # (bsz)
        
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        
        next_token = torch.where(is_finished, pad_token_id, next_token)
        
        tokens[:, cur_pos] = next_token
        
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        
        prev_pos = cur_pos
        if is_finished.all():
            break
    
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()
    
    episodes = []
    for i in range(bsz//num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]):]
            
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[:generated_token_ids.index(pad_token_id)]
            
            generated_text = tokenizer.decode(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers = batch.numbers[i],
                target = batch.target[i],
                end_token = end_token,
            )
            
            episode = Episode(
                prefix = batch.prefix[i],
                text = batch.prefix[i] + generated_text,
                prefix_token_ids = batch.prefix_token_ids[i],
                prefix_tokens = batch.prefix_tokens[i],
                generated_token_ids = generated_token_ids,
                is_finished = is_finished_list[idx],
                reward = rewards["reward"], 
                reward_info = rewards["reward_info"],
            )
            episodes.append(episode)
    
    print("\r", end=" " * 100, flush=True)
    return episodes

def normalize_rewards_per_group(episodes: List[Episode]):
    """
    Normalize the rewards per group

    Args:
        episodes (List[Episode]): the list of episodes
    
    Returns:
        List[Episode]: the list of episodes with normalized rewards
    """
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    
    output = []
    for group in groups.values():
        group_rewards =[item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-6)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    
    return output   

def compute_entropy(logits: torch.Tensor):
    """
    Compute the entropy of the logits

    Args:
        logits (torch.Tensor): the logits of the model without softmax, shape is (batch_size, seq_len, sqevocab_size)
    
    Returns:
        torch.Tensor: the entropy of each position, shape is (batch_size, seq_len)
    """
    probs = nn.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs*logits, dim=-1)
    return entropy

def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Use GRPO to update the policy

    Args:
        model (Transformer): the pretrained model to update the policy
        optimizer (torch.optim.Optimizer): the optimizer to update the policy
        episodes (List[Episode]): the list of episodes to update the policy
        micro_batch_size (int): the size of micro batch
        pad_token_id (int): the pad token id
        max_grad_norm (float): the maximum gradient norm
        device (torch.device): the device to run the model
        dtype (torch.dtype): the data type to run the model
    
    Returns:
        dict: the dictionary of the loss, gradient, and entropy
    """
    # Normalize the rewards per group
    episodes = normalize_rewards_per_group(episodes)
    
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    
    # Calculate the sum of target tokens
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    
    entropy = 0.0 
    
    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        
        # Confirm the start and end index of the micro batch
        j = min(i + micro_batch_size, len(episodes))

        batch_episodes = episodes[i:j]
        
        batch_length =[
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        
        batch_max_length = max(batch_length)
        
        # Create the batch token ids with padding
        batch_token_ids  = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_length[i])
            for i, episode in enumerate(batch_episodes)
        ]
        
        # Create the batch masks to label the generated tokens
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_length[i])
            for i, episode in enumerate(batch_episodes)
        ]
       
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=device)
        batch_masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
        batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=device)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            # Input the token ids and remove the last token as the target
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()
        
        # Calculate the log prob
        log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.size(0), -1)
        
        with torch.no_grade():
            token_entropy = compute_entropy(logits)
            entropy += (token_entropy * target_masks).sum() / num_target_tokens
        
        # Calculate the objective functions: log_prob * advantages
        obj = log_probs * batch_advantages[:None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()
    
    grade_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    return {
        "loss": loss.item(),
        "grade_norm": grade_norm.item(),
        "entropy": entropy.item(),
    }