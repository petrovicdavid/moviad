from abc import abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Optional

from torch.utils.data.dataset import Dataset
from moviad.utilities.configurations import TaskType, Split
from moviad.datasets.dataset_arguments import DatasetArguments

class VADDataset(Dataset):
    """
    Args:
        split (Split): The split of the dataset (train, val, test).
        dataset_path (str): The path to the dataset.
        contamination_ratio (float): The ratio of contamination in the dataset.
    """

    #TODO: to be defined the splitting strategy for all the VAD datasets
    #TODO: add the contamination part

    def __init__(
        self,
        arguments: DatasetArguments
    ):
        self.dataset_arguments = arguments

    @abstractmethod
    def load_dataset(self): ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod 
    def split_dataset(self, train_size, valid_size): ...

    @staticmethod
    def get_categories() -> list: ...

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def __getitem__(self, index): ...

    @abstractmethod
    def contaminate(self, ratio: float, seed: int = 42) -> int: ...

    @abstractmethod
    def compute_contamination_ratio(self) -> float: ...
