import math

import torch
from torch.optim import AdamW


class MemoryEfficientAdamW(AdamW):
    """
    MemoryEfficientAdamW is a memory-efficient AdamW optimizer implementation.
    It keeps parameters and gradients on GPU, but stores optimizer states on CPU when enabled.
    When disabled, it behaves exactly like the standard AdamW optimizer.
    
    Args:
        params (iterable): Parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate. Default: 1e-3.
        betas (Tuple[float, float], optional): Decay rates for first and second moment estimates. Default: (0.9, 0.999).
        eps (float, optional): Small constant added to denominator for numerical stability. Default: 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty). Default: 1e-2.
        amsgrad (bool, optional): Whether to use AMSGrad variant. Default: False.
        pin_memory (bool, optional): Whether to pin optimizer state tensors to memory. Default: True.
        enabled (bool, optional): Whether to enable memory-efficient mode. Default: True.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
    eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        pin_memory=True,
        enabled=True,
    ):
        # Call parent AdamW initialization method with necessary parameters
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        # Whether to pin optimizer state tensors to memory
        self.pin_memory = pin_memory
        # Whether to enable memory-efficient mode
        self.enabled = enabled

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure (callable, optional): A callable that can re-evaluate the model and return the loss. Default: None.
        
        Returns:
            loss (torch.Tensor, optional): Returns loss value if closure is provided.
        """
        if not self.enabled:
            return super(MemoryEfficientAdamW, self).step(closure)

        loss = None
        if closure is not None:
            # If closure is provided, execute it with gradients enabled
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Initialize lists to store parameters, gradients, first moments, second moments, max second moments, and step counts
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            # Get decay rates for first and second moment estimates of current parameter group
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue  # Skip if parameter has no gradient

                params_with_grad.append(p)
                grads.append(p.grad)

                # Initialize state dictionary if not already initialized
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Store optimizer states on CPU and pin to memory (if enabled)
                    device = "cpu"
                    pin_memory = self.pin_memory
                    dtype = torch.float32

                    state["exp_avg"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, device=device, pin_memory=pin_memory, dtype=dtype
                    )
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p.data, device=device, pin_memory=pin_memory, dtype=dtype
                        )

                # Get values from current parameter's state
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # Update step count
                state["step"] += 1
                state_steps.append(state["step"])

            # Perform memory-efficient update for all parameters in current parameter group
            self._memory_efficient_update(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def _memory_efficient_update(
        self,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad,
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
    ):
        """
        Perform AdamW parameter updates on GPU while keeping optimizer states stored on CPU.
        Uses pinned memory for efficient CPU to GPU transfer of optimizer states.
        
        Args:
            params (List[torch.Tensor]): List of parameters to update.
            grads (List[torch.Tensor]): List of gradients for corresponding parameters.
            exp_avgs (List[torch.Tensor]): List of first moment estimates.
            exp_avg_sqs (List[torch.Tensor]): List of second moment estimates.
            max_exp_avg_sqs (List[torch.Tensor]): List of max second moment estimates (if amsgrad enabled).
            state_steps (List[int]): List of step counts for each parameter.
            amsgrad (bool): Whether to use AMSGrad variant.
            beta1 (float): Decay rate for first moment estimate.
            beta2 (float): Decay rate for second moment estimate.
            lr (float): Learning rate.
            weight_decay (float): Weight decay.
            eps (float): Small constant for numerical stability.
        """
        for i, param in enumerate(params):
            grad = grads[i]
            param_device = param.device

            # Access optimizer states - efficient transfer due to pinned memory
            exp_avg = exp_avgs[i].to(param_device, non_blocking=True)
            exp_avg_sq = exp_avg_sqs[i].to(param_device, non_blocking=True)

            step = state_steps[i]

            # Decay first and second moment estimates
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            if amsgrad:
                # Access max_exp_avg_sq - efficient transfer due to pinned memory
                max_exp_avg_sq = max_exp_avg_sqs[i].to(param_device, non_blocking=True)
                # Maintain maximum of all second moment estimates so far
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                denom = max_exp_avg_sq.sqrt().add_(eps)
                # Store maximum value back to CPU
                max_exp_avg_sqs[i].copy_(max_exp_avg_sq, non_blocking=True)
            else:
                denom = exp_avg_sq.sqrt().add_(eps)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            # Apply weight decay directly to parameters (AdamW)
            if weight_decay != 0:
                param.mul_(1 - lr * weight_decay)

            # Update parameters (directly on GPU)
            param.addcdiv_(exp_avg, denom, value=-step_size)

            # Store optimizer states back to CPU
            exp_avgs[i].copy_(exp_avg, non_blocking=True)
            exp_avg_sqs[i].copy_(exp_avg_sq, non_blocking=True)