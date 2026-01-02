from abc import abstractmethod
import torch
import torch.nn as nn

from moviad.models.training_args import TrainingArgs

class VADModel(nn.Module):

    @abstractmethod
    def forward(self, batch: torch.Tensor): ... 

    def __call__(self, batch: torch.Tensor):
        self.forward(batch)

    def parameters(self): ...

    @abstractmethod
    def train_epoch(self, epoch: int, train_dataloader: torch.utils.data.DataLoader, device: torch.device, training_args: TrainingArgs): ...

    @abstractmethod
    def train_step(self, batch: torch.Tensor, device: torch.device, training_args: TrainingArgs): ...