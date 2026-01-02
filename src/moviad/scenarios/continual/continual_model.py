from abc import abstractmethod, ABC

from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

class ContinualModel(ABC):

    def __init__(self, model: VADModel, train_args: TrainingArgs):
        self.model = model
        self.train_args = train_args

    @abstractmethod
    def start_task(self): ...

    @abstractmethod
    def train_task(self): ...

    @abstractmethod
    def end_task(self): ...
