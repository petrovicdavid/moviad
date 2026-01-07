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

    def __to_dict__(self):
        return {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "evaluation_epoch_interval": self.evaluation_epoch_interval,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
        }

    def init_train(self, model: torch.nn.Module):
        pass