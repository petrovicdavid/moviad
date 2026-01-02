from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer

from tqdm import trange

class FineTuning(ContinualModel):

    def __init__(self, model: VADModel):
        super().__init__(model)

    def start_task(self):
        pass

    def train_task(self, task_index: int, train_dataset, test_dataset, train_args, metrics, device, logger):

        trainer = Trainer(
            train_args,
            self.vad_model,
            train_dataset,
            test_dataset,
            metrics=metrics,
            device=device,
            logger=None,
            logging_prefix=f"Task_T{task_index}/",
        )

        trainer.train()
        

    def end_task(self):
        pass
