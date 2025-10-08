from typing import Union, Dict, Callable, Tuple, Optional
import torch
import matplotlib.pyplot as plt
import math


class BaseScheduler:
    def __call__(self, losses: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        raise NotImplementedError("Scheduler must implement __call__ method.")


class ScalarScheduler:
    """
    A scalar scheduler for dynamically adjusting values (e.g., loss weights)
    over epochs using warmup and annealing.

    Args:
        start (float): Initial value.
        end (float): Final target value.
        annealing_steps (int): Total steps for annealing (including warmup).
        warmup_steps (int): Number of warmup steps before annealing starts.
        mode (str): Scheduling mode, modes supported:
            - linear
            - cosine
            - exponential
            - constant
    """
    def __init__(
        self,
        start: float,
        end: float,
        annealing_steps: int,
        warmup_steps: int = 0,
        mode: str = "linear",
    ):
        self.start = start
        self.end = end
        self.annealing_steps = annealing_steps
        self.warmup_steps = warmup_steps
        self.mode = mode.lower()

    def __call__(self, epoch: int) -> float:
        # Warmup phase
        if epoch < self.warmup_steps:
            return self.start * (epoch / self.warmup_steps)
        
        # Annealing phase
        progress = min(
            (epoch - self.warmup_steps) / max(1, self.annealing_steps - self.warmup_steps), 
            1.0
        )

        if self.mode == "linear":
            scale = 1 - progress
        
        elif self.mode == "cosine":
            scale = (1 + math.cos(math.pi * progress)) / 2

        elif self.mode == "exponential":
            beta = 5.0 
            smin = math.exp(-beta)
            _scale = math.exp(-beta * (1 - progress))
            scale = 1 - (_scale - smin) / (1 - smin)

        elif self.mode == "constant":
            scale = 1
        
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented.")
    
        return self.start * scale + self.end * (1 - scale)
    
    def plot(self, total_epochs: int = None, log_scale: bool = False):
        """
        Plot the value schedule over epochs.

        Args:
            total_epochs (int): Total number of epochs to visualize. 
                                If None, uses self.annealing_steps.
        """
        if total_epochs is None:
            total_epochs = self.annealing_steps

        epochs = list(range(total_epochs))
        values = [self(epoch) for epoch in epochs]

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, values, label=f"{self.mode.capitalize()} Schedule")
        plt.axvline(self.warmup_steps, color='gray', linestyle='--', label='Warmup End')
        plt.axvline(self.annealing_steps, color='red', linestyle='--', label='Annealing End')
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("ScalarScheduler Value Schedule")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if log_scale:
            plt.yscale('log')
        plt.show()


class CompositeLossScheduler:
    """
    Combines multiple per-loss schedulers and fixed weights into one unified scheduler.
    """
    def __init__(
        self,
        schedulers: Dict[str, BaseScheduler],      # e.g. {"mmd": scheduler1, "physics": scheduler2}
        fixed_weights: Dict[str, float] = None,    # e.g. {"recon": 1.0, "pred": 0.1}
        disabled_losses: list[str] = None,         # e.g. ["recon", "pred"] (not used in this implementation, but can be added for future use)
        log_fn=None                                # Optional logging function, not used in this implementation
    ):
        self.schedulers = schedulers
        self.fixed_weights = fixed_weights or {}
        self.disabled_losses = set(disabled_losses or [])
        self.log_fn = log_fn

    def __call__(self, losses: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]: 
        weights = {}

        for name in losses:
            if name in self.disabled_losses:
                continue  # Skip disabled losses

            if name in self.fixed_weights: # Use fixed weights if available
                weights[name] = self.fixed_weights[name]
            elif name in self.schedulers:  # Use the scheduler for this loss
                weight_dict = self.schedulers[name](losses, epoch)
                weights[name] = weight_dict[name]
            else:
                raise ValueError(f"Loss '{name}' has no assigned scheduler or fixed weight.")
        
        if self.log_fn is not None:
            self.log_fn(f"[Epoch {epoch}] Loss Weights: {weights}")

        return weights
    
    def disable_loss(self, name: str):
        """
        Disable a specific loss term from being scheduled or used.
        """
        self.disabled_losses.add(name)

    def enable_loss(self, name: str):
        """
        Re-enable a previously disabled loss term.
        """
        self.disabled_losses.discard(name)


class MagnitudeMatchScheduler(BaseScheduler):
    """
    A dynamic loss weighting scheduler that adjusts the relative weights of multiple loss terms
    based on their magnitudes, to match a target ratio with respect to a reference loss.
    
    Key Features:
    - Adjusts each loss weight based on the ratio between its current magnitude and the reference loss.
    - Optionally applies Exponential Moving Average (EMA) for smoother adjustment.
    - Clamps the maximum allowed weight to `lambda_max`.
    - Supports callable `target_ratios` for time-dependent weight scheduling.
    - Tracks weight history for visualization.

    Args:
        loss_names (List[str]): Names of all loss terms to be scheduled.
        reference_loss_name (str): Name of the reference loss used for normalization.
        target_ratios (Dict[str, Union[float, Callable]]): Desired target ratio relative to the reference loss.
        use_ema (bool): Whether to apply EMA smoothing to loss magnitudes.
        ema_alpha (float): EMA decay factor (closer to 1 = slower smoothing).
        lambda_max (float): Upper bound on any loss weight to avoid explosion.
    """

    def __init__(
        self, 
        loss_names: list[str],                      
        reference_loss_name: str,                  
        target_ratios: Dict[str, Union[float, Callable[[int], float]]], 
        use_ema: bool = True,
        ema_alpha: float = 0.9,
        lambda_max: float = 1.0,
    ):

        if reference_loss_name not in loss_names:
            raise ValueError("Reference loss must be in loss_names.")
        
        target_ratios[reference_loss_name] = 1.0 
        for name in loss_names:
            if name not in target_ratios: # Check if target ratio is specified
                raise ValueError(f"No target_ratio specified for loss '{name}'.")

        self.loss_names = loss_names
        self.reference_loss_name = reference_loss_name
        self.target_ratios = target_ratios
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.lambda_max = lambda_max

        # Initialize EMA trackers
        self.ema = {name: None for name in loss_names}

        # Store weight history for visualization
        self.weight_history = {name: [] for name in loss_names}

    def _ema_update(self, name: str, value: float) -> float:
        """Update and return the exponential moving average (EMA) of a value."""
        if self.ema[name] is None: # Initialize EMA if not set
            self.ema[name] = value
        else: 
            self.ema[name] = self.ema_alpha * self.ema[name] + (1 - self.ema_alpha) * value
        return self.ema[name]

    def __call__(self, losses: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """
        Compute and return the adjusted loss weights for this epoch.

        Args:
            losses (Dict[str, torch.Tensor]): Current loss values.
            epoch (int): Current epoch number (for dynamic ratio scheduling).

        Returns:
            Dict[str, float]: Updated weights for each loss term.
        """
        weights = {}

        # Get reference loss magnitude
        ref_value = losses[self.reference_loss_name].detach().item()
        if self.use_ema:
            ref_value = self._ema_update(self.reference_loss_name, ref_value)

        for name in self.loss_names:
            if name == self.reference_loss_name: # Reference loss weight is fixed to 1.0
                weight = 1.0  # Reference loss weight is fixed
            else:
                if name not in losses: # Check if loss exists
                    raise ValueError(f"Loss '{name}' is not in current losses dict.")

                val = losses[name].detach().item()
                if self.use_ema:
                    val = self._ema_update(name, val)

                # Get scheduled target ratio for this loss (could be callable or float)
                if callable(self.target_ratios[name]):
                    target_ratio = self.target_ratios[name](epoch)
                else:
                    target_ratio = self.target_ratios[name]

                # Compute dynamic weight
                if val == 0:
                    weight = self.lambda_max
                else:
                    weight = target_ratio * (ref_value / val)
                    weight = min(weight, self.lambda_max)

            weights[name] = weight
            self.weight_history[name].append(weight)

        return weights
    
    def plot_weight_history(self):
        """Plot the evolution of each loss weight over epochs."""
        if not any(len(hist) > 0 for hist in self.weight_history.values()):
            print("No weight history to plot.")
            return

        plt.figure(figsize=(8, 5))
        for name, history in self.weight_history.items():
            plt.plot(history, label=name)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Weight")
        plt.title("Loss Weight Scheduling over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
