from abc import abstractmethod, ABC
from moviad.models.vad_model import VADModel

class ContinualModel(ABC):

    def __init__(self, model: VADModel):
        self.model = model

    @abstractmethod
    def start_task(self): ...

    @abstractmethod
    def train_task(self): ...

    @abstractmethod
    def end_task(self): ...
