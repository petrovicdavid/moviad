from abc import abstractmethod, ABC

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

class ContinualModel(ABC):

    def __init__(self, vad_model: VADModel):
        self.vad_model = vad_model

    @abstractmethod
    def start_task(self, train_args: TrainingArgs): ...

    @abstractmethod
    def train_task(self, task_index: int, train_dataset, test_dataset, train_args, metrics, device, logger): ...

    @abstractmethod
    def end_task(self): ...
