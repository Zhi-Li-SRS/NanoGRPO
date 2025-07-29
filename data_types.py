from dataclasses import dataclass 
from typing import Dict, List, Optional

@dataclass 
class Episode:
    """
    Episode class is used to store all of training or inference data for one episode
    
    Args:
        prefix (str): 
            - the prefix of the episode, which is the initial input of the episode
        
        text (str): 
            - the generated response of the episode
        
        prefix_token_ids (List[int]): 
            - the token ids of the prefix
        
        prefix_tokens (List[str]): 
            - the tokens of the prefix
        
        generated_token_ids (List[int]): 
            - the token ids of the generated response which is used to calculate the loss
        
        is_finished (bool): 
            - whether the episode is finished
        
        reward (float): 
            - the reward of the episode
        
        reward_info (Dict[str, float]): 
            - more reward info of the episode, key is the related name of the reward, value is the reward score
    """
    prefix: str
    text: str 
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]

@dataclass 
class MiniBatch:
    """
    MiniBatch class is used to store multiple episodes data in a mini-batch
    
    Args:
        prefix (List[str]): 
            - all of the prefix of the current mini-batch
        
        prefix_tokens (List[List[str]]): 
            - all of the prefix tokens of the current mini-batch
        
        prefix_token_ids (List[List[int]]): 
            - all of the prefix token ids of the current mini-batch
        
        numbers (List[List[int]]): 
            - all of the numbers of the current mini-batch such as time-steps, position, etc.
        
        target (List[int]): 
            - all of the target of the current mini-batch such as the next predicted token id in LLM
    """
    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]]
    target: List[int]
