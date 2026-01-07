from abc import abstractmethod
import torch
import torch.nn as nn

from moviad.models.training_args import TrainingArgs

class VADModel(nn.Module):

    @abstractmethod
    def forward(self, batch: torch.Tensor): ... 

    @abstractmethod
    def parameters(self): ...

    @abstractmethod
    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader, training_args: TrainingArgs): ...

    @abstractmethod
    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs): ...