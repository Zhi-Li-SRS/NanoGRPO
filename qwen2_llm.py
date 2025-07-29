import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import torch 
import torch.nn.functional as F
from torch import nn 

@dataclass
class Qwen2Config:
    """
    Qwen2Config class is used to store the configuration of the Qwen2 model
    
    Args:
        attention_dropout (float): the dropout probability of the attention mechanism. default is 0.0.
        bos_token_id (int): the id of the beginning of sequence. default is 151643.
        eos_token_id (int): the id of the end of sequence. default is 151645.
        hidden_act (str): the type of the hidden layer activation function. default is "silu" (Sigmoid Linear Unit).
        hidden_size (int): the size of the hidden layer. default is 2048.
        initializer_range (float): the standard deviation range of the weight initialization. default is 0.02.
        intermediate_size (int): the size of the intermediate layer. default is 11008.
        max_position_embeddings (int): the maximum length of the position encoding. default is 32768.
        max_window_layers (int): the maximum number of the window layers. default is 70.
        model_type (str): the type of the model. default is "qwen2".
        num_attention_heads (int): the number of the attention heads. default is 16.
        num_hidden_layers (int): the number of the hidden layers. default is 36.
        num_key_value_heads (int): the number of the key value attention heads. default is 2.
        rms_norm_eps (float): the small constant in the RMS normalization to prevent division by zero. default is 1e-6.
        rope_theta (float): the parameter θ in the RoPE (Rotary Position Embedding). default is 1000000.0.
        sliding_window (int): the size of the sliding window. default is 32768.
        tie_word_embeddings (bool): whether to tie the word embeddings. default is True.
        torch_dtype (str): the data type of the PyTorch tensor. default is "bfloat16".
        use_cache (bool): whether to use the cache. default is True.
        use_sliding_window (bool): whether to use the sliding window. default is False.
        vocab_size (int): the size of the vocabulary. default is 151936.
    """
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768
    max_window_layers: int = 70
    model_type: str = "qwen2"
    num_attention_heads: int = 16
    num_hidden_layers: int = 36
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936


class RMSNorm(nn.Module):
    """
    RMSNorm class is used to implement the RMSNorm normalization
    
    Args:
        dim (int): the dimention of the input data
        eps (float): the small constant in the RMS normalization to prevent division by zero. default is 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        forward pass of the RMSNorm
        
        Args:
            x (torch.Tensor): the input tensor
        
        Returns:
            torch.Tensor: the normalized tensor
        """
        input_type = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_type)
        return x


def rotate_half(x):
    """
    Rotate the half of the tensor for the RoPE (Rotary Position Embedding)
    Args:
        x (torch.Tensor): the input tensor, shape is (..., dim)
    
    Returns:
        torch.Tensor: the rotated tensor, shape is (..., dim)
    """
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """
    Apply RoPE to q and k tensors
    
    Args:
        q (torch.Tensor): the query tensor, shape is (..., seq_len, dim)
        k (torch.Tensor): the key tensor, shape is (..., seq_len, dim)
        cos (torch.Tensor): the cosine tensor, shape is (..., seq_len, dim)
        sin (torch.Tensor): the sine tensor, shape is (..., seq_len, dim)
        unsqueeze_dim (int): the dimension to unsqueeze, default is 2
    
    Returns:
        tuple:
            - q_embed (torch.Tensor): the rotated query tensor, shape is (..., seq_len, dim)  
            - k_embed (torch.Tensor): the rotated key tensor, shape is (..., seq_len, dim)
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """
    Multi-head attention mechanism for the Qwen2 model

    Args:
        args (Qwen2Config): the configuration of the Qwen2 model
    """
    def __init__(self, args: Qwen2Config):
        super().__init__()
        
        self.n_kv_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = self.n_kv_heads 
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=True
        )
        
        self.k_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=True
        )
        
        self.v_proj = nn.Linear(
            args.hidden_size,
            self.n_kv_heads * self.head_dim,
            bias=True
        )
        
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False
        )
        
        self.args = args
    
    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Init kv cache to speed up the inference

        Args:
            max_batch_size (int): the maximum batch size    
            max_seq_len (int): the maximum sequence length
            dtype (torch.dtype): the data type of the tensor
            device (torch.device): the device of the tensor
        """
        # Define the shape of the kv cache (batch_size, seq_len, n_kv_heads, head_dim)
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)
    
    def del_kv_cache(self):
        """Delete the kv cache to release the memory"""
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        """
        Forward pass of the Attention
        
        Args:
            x (torch.Tensor): the input tensor, shape is (batch_size, seq_len, hidden_size)
            pos_embed (Tuple[torch.Tensor, torch.Tensor]): the position embedding, shape is (2, seq_len, head_dim)
            start_pos (Optional[Union[int, torch.Tensor]]): the start position of the sequence used for inference, default is None
        """
        bsz, seq_len, _  = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        
        cos, sin = pos_embed
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if start_pos is not None:
            # Inference mode
            end_pos = start_pos + seq_len
            
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            
            output = F.scaled_dot_product_attention(
                query = xq.transpose(1, 2), 
                key = self.cache_k[:bsz, :end_pos].transpose(1, 2),
                value = self.cache_v[:bsz, :end_pos].transpose(1, 2),
                is_causal = True if seq_len > 1 else False,
                enable_gqa = True,
            ).transpose(1, 2) # (bsz, n_heads, seq_len, head_dim)
        
        else:
            output = F.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,   
                enable_gqa=True,
            ).transpose(1, 2)
        
        output = output.reshape(bsz, seq_len, -1)
        return self.o_proj(output) # (bsz, seq_len, hidden_size)

class FeedForward(nn.Module):
    """
    FFN parts in transformer

    Args:
        dim (int): the dimension of input and output
        intermediate_size (int): the dimension of the intermediate layer

    """
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the FeedForward
        
        Args:
            x (torch.Tensor): the input tensor, shape is (batch_size, seq_len, hidden_size)
        
        Returns:
            torch.Tensor: the output tensor, shape is (batch_size, seq_len, hidden_size)
        """
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)

class TransformerBlock(nn.Module):
    """
    The basic block of the Transformer model including the attention and FFN

    Parameters:
       layer_id (int): The number of the layer in the transformer
       args (Qwen2Config): the configuration of the Qwen2 model
    """
    
    def __init__(self, layer_id: int, args:Qwen2Config):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(args.hidden_size, args.intermediate_size)
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, args.rms_norm_eps)
    
    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        """
        Forward pass of the TransformerBlock
        
        Params:
            x (torch.Tensor): the input tensor, shape is (batch_size, seq_len, hidden_size)
            pos_embed (Tuple[torch.Tensor, torch.Tensor]): the position embedding, shape is (2, seq_len, head_dim)
            start_pos (Optional[Union[int, torch.Tensor]]): the start position of the sequence used for inference, default is None
        """
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for the Qwen2 model
    
    Args:
        config (Qwen2Config): the configuration of the Qwen2 model
        device (torch.device): the device of the model
    """
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        
        # Every attention head dim
        dim = config.hidden_size // config.num_attention_heads
        
        # Use hybrid flop to calculate the cos and sin
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """
        Forward pass of the RotaryEmbedding
        
        Args:
            x (torch.Tensor): the input tensor, shape is (batch_size, seq_len, hidden_size)
            pos (torch.Tensor): the position tensor, shape is (batch_size, seq_len)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the cos and sin of the position embedding, shape is (batch_size, seq_len, head_dim)
        """
        inv_freq = self.inv_freq[None, :, None].float().expand(pos.shape[0], -1, 1)
        pos = pos[:, None, :].float()
        device_type = x.device.type
        
        # Forbid the autocast
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float().to(x.device) @ pos.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1) # (batch_size, seq_len, head_dim)
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Transformer(nn.Module):
    """
    This Transformer model realized the Qwen2LLM
    
    Args:
        config (Qwen2Config): the configuration of the Qwen2 model
        device (torch.device): the device of the model
    """
    def __init__(self, config: Qwen2Config, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.vocab_size = config.vocab_size
        self.n_layers = config.num_hidden_layers
        
        # token embedding vocab_size -> hidden_size
        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size)
        
        with torch.device(device):
            self.rotary_emb = Qwen2RotaryEmbedding(config, device)
        
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))
        
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        if not config.tie_word_embeddings: 
            self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
    
    def output_project(self, x: torch.Tensor):
        """
        Project the output to the vocab_size

        Args:
            x (torch.Tensor): the output tensor, shape is (batch_size, seq_len)
        
        Returns:
            torch.Tensor: the output tensor, shape is (batch_size, seq_len, vocab_size)
        """
        if self.config.tie_word_embeddings:
            return x @ self.embed_tokens.weight.T
        else:
            return self.lm_head(x)
    
    def forward(self, tokens: torch.Tensor):
        """
        Implement the forward pass of the Qwen2LLM
        
        Args:
            tokens (torch.Tensor): the input tensor, shape is (batch_size, seq_len)
        
        Returns:
            torch.Tensor: the output tensor, shape is (batch_size, seq_len, vocab_size)
        """
        bsz, seq_len = tokens.shape
        h = self.embed_tokens(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device, dtype=torch.int32) # (seq_len)
        pos_embed = self.rotary_emb(h, pos[None, :]) # (bsz, seq_len, hidden_size)
        
        pipe = []
        for layer in self.layers:
            pipe.append(lambda x, layer=layer: layer(x, pos_embed))
        
        pipe.append(self.norm.forward)
        pipe.append(self.output_project)
        
        return torch.utils.checkpoint.checkpoint_sequential(
            pipe, len(pipe), h, use_reentrant=False
        )
    
    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        """
        Inference mode

        Args:
            tokens (torch.Tensor): the input tensor, shape is (batch_size, seq_len)
            start_pos (Union[int, torch.Tensor]): the start position of the sequence used for inference
        
        Returns:
            torch.Tensor: the output tensor, shape is (batch_size, 1, vocab_size)
        """
        bsz, seq_len = tokens.shape
        del bsz
        
        h = self.embed_tokens(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device, dtype=torch.int32)[None, :]
        if isinstance(start_pos, torch.Tensor):
            pos = start_pos[:, None] + pos
        
        else:
            pos.add_(start_pos)
        
        pos_embed = self.rotary_emb(h, pos)
        
        for layer in self.layers:
            h = layer(h, pos_embed, start_pos)
        
        h = h[:, -1:, :]
        h = self.norm(h)
        return self.output_project(h)

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """
        Initialize the kv cache
        
        Args:
            max_batch_size (int): the maximum batch size
            max_seq_len (int): the maximum sequence length
            device (torch.device): the device of the model
            dtype (torch.dtype): the data type of the tensor
        """
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len, dtype, device)
    
    def del_kv_cache(self):
        """Delete the kv cache to release the memory"""
        for layer in self.layers:
            layer.self_attn.del_kv_cache()
    

    @classmethod
    def from_pretrained(cls, ckpt_path:str, device:torch.device):
        
        config_file = Path(ckpt_path) / "config.json"
        with open(config_file, "r") as f:
            config = json.load(f)
        args = Qwen2Config(
            attention_dropout=config["attention_dropout"],
            bos_token_id=config["bos_token_id"],
            eos_token_id=config["eos_token_id"],
            hidden_act=config["hidden_act"],
            hidden_size=config["hidden_size"],
            initializer_range=config["initializer_range"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
            max_window_layers=config["max_window_layers"],
            model_type=config["model_type"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            vocab_size=config["vocab_size"],
            rms_norm_eps=config["rms_norm_eps"],
            rope_theta=config["rope_theta"],
            sliding_window=config["sliding_window"],
            use_sliding_window=config["use_sliding_window"],
            use_cache=config["use_cache"],
            tie_word_embeddings=config["tie_word_embeddings"],
            torch_dtype=config["torch_dtype"],
        )
    
        with torch.device("meta"):
            model = cls(args, device)
        
        import safetensors.torch
        
        model_weight_files = sorted(Path(ckpt_path).glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        
        weights = {k.replace("model.", ""): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=True, assign=True)
        return model.to(device)