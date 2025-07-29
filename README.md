# NanoGRPO: A Minimal Implementation of Group Relative Policy Optimization

A lightweight, efficient implementation of Group Relative Policy Optimization (GRPO) for fine-tuning language models on simple mathematical reasoning tasks.

## Overview

NanoGRPO implements GRPO, a reinforcement learning algorithm designed to improve language model performance through policy optimization. This implementation focuses on training models to solve arithmetic problems, where the model must create mathematical equations using given numbers to reach a target value.

## Features

- **Memory-Efficient Training**: Custom AdamW optimizer with CPU state storage to reduce GPU memory usage
- **Qwen2 Model Support**: Full implementation of Qwen2 transformer architecture with KV-cache optimization
- **GRPO Algorithm**: Complete rollout, reward computation, and policy update pipeline
- **Gradient Checkpointing**: Memory-efficient training through activation checkpointing

## Project Structure

```
NanoGRPO/
├── grpo.py           # Core GRPO algorithm implementation
├── optimizer.py      # Memory-efficient AdamW optimizer
├── qwen2_llm.py     # Qwen2 transformer model implementation
├── tokenizer.py     # Tokenizer with chat template support
├── math_task.py     # Simple math task and reward functions
├── data_types.py    # Data structures for episodes and batches
└── train.py         # Training script 
```

## Key Components

### GRPO Algorithm (`grpo.py`)
- **Rollout Generation**: Generate model responses with sampling
- **Reward Normalization**: Per-group reward normalization for stable training
- **Policy Updates**: GRPO loss computation and gradient updates
- **Entropy Computation**: Policy entropy for regularization

### Memory-Efficient Optimizer (`optimizer.py`)
- Stores optimizer states on CPU while keeping parameters on GPU
- Reduces GPU memory usage significantly during training
- Supports standard AdamW features including weight decay and AMSGrad

### Qwen2 Implementation (`qwen2_llm.py`)
- Complete Qwen2 transformer architecture
- Rotary Position Embedding (RoPE)
- Grouped Query Attention (GQA)
- KV-cache for efficient inference
- Gradient checkpointing support


## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/NanoGRPO.git
cd NanoGRPO

# Install dependencies
pip install torch safetensors tokenizers jinja2 pandas numpy
```

### Simple Math Task Format

The model learns to solve problems like:
```
Input: "Using the numbers [1, 2, 3, 4], create an equation that equals 10."
Expected Output: 
<think>
I need to find a way to combine 1, 2, 3, 4 to get 10.
Let me try: (1 + 2 + 3) * 4 = 6 * 4 = 24, too big.
How about: 1 + 2 + 3 + 4 = 10. Perfect!
</think>
<answer>1 + 2 + 3 + 4</answer>
```

## License

This project is open source and available under the [MIT License](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{nanogrpo2024,
  title={NanoGRPO: A Minimal Implementation of Group Relative Policy Optimization},
  author={Zhi Li},
  year={2025},
  url={https://github.com/Zhi-Li-SRS/NanoGRPO}
}
```
## Acknowledgments
- Based on the GRPO algorithm for policy optimization from DeepSeek
- Qwen2 model architecture by Alibaba Cloud
- Inspired by memory-efficient training techniques for large language models