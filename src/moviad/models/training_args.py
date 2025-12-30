from dataclasses import dataclass
from typing import Callable
import torch

@dataclass
class TrainingArgs: 
    batch_size: int
    epochs: int
    evaluation_epoch_interval: int = 1 
    optimizer: torch.optim.Optimizer | None = None
    loss_function: Callable | None = None
