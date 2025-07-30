import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from math_task import CountdownTasksDataset, reward_function
from grpo import rollout, update_policy
from optimizer import MemoryEfficientAdamW
from qwen2_llm import Qwen2Config, Transformer
from tokenizer import Tokenizer

def evaluate(model, tokenizer, device, dtype, config):
    """
    Evaluate the performace of the model on the task

    Parameters
    ----------
    model: the model instance need to be evaluated
    tokenizer: the tokenizer instance to translate the text to token ids
    device: the device to run the model on 
    dtype: the data type, such as bfloat16 or float32
    config (dict): the configuration of the model including:
        - "data" : the data configuration
            - "path" (str): the path to the data
            - "test_size" (int): the size of the test set
        - "training": the training configuration
            - "batch_size" (int): the batch size
            - "max_gen_len" (int): the maximum length of the generated text
    
    Returns:
        float: the reward score of the average answers
    """
    
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    
    # Create a generator to generate the random number
    generator = torch.Generator(device=device)
    
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator, 
        drop_last=False
    )
    
    success = []
    
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])


    return np.mean(success)

def main(config_path: str):
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    
    torch.random.manual_seed(config["training"]["seed"])
    
    batch_size = config["training"]["batch_size"]
    num_questions_per_batch = config["training"]["num_questions_per_batch"]
    num_answers_per_question = batch_size // num_questions_per_batch
    
    current_time = datetime.now().strftime(r"%Y%m%d_%H%M%S")
    tb_writer = SummaryWriter(
        log_dir=f"{config["training"]["log_dir"]}/{current_time}"   
    )
    
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    
    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],   
        tokenizer=tokenizer,   
        split="train",    
        test_size=config["data"]["test_size"],   
    )
  
    generator = torch.Generator(device=device)  
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,    
        collate_fn=CountdownTasksDataset.collate_fn,  
        generator=generator,   
        batch_size=num_questions_per_batch,   
    )
    
    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()
    
    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr = config["training"]["learning_rate"],
        weight_decay = config["training"]["weight_decay"],
        betas = config["training"]["betas"],
        enabled = config["training"]["memory_efficient_adamw"],
    )
    
    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=num_answers_per_question,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        results = update_policy(
            model=model,
            optimizer=optimizer,
            episodes=episodes,
            micro_batch_size=config["training"]["micro_batch_size"],
            pad_token_id=tokenizer.pad_token_id,
            max_grad_norm=config["training"]["max_grad_norm"],
            entropy_coef=config["training"]["entropy_coef"],
            device=device,
            dtype=dtype,
        )
        
        torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time
        
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"]
            for episode in episodes
        ]
        
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
     
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]

        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )
        

        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)

        for i, episode in enumerate(episodes):
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)
            
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)