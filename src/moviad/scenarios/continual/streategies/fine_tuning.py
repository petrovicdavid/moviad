from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer

from tqdm import trange

class FineTuning(ContinualModel):

    def __init__(self, model: VADModel, train_args: TrainingArgs):
        super().__init__(model, train_args)

    def start_task(self):
        pass

    def train_task(self, train_dataset, test_dataset, metrics, device, logger):

        trainer = Trainer(
            self.train_args,
            self.model,
            train_dataset,
            test_dataset,
            metrics=metrics,
            device=device,
            logger=logger,
            save_path=None,
            saving_criteria=None, # NB: no saving criteria for continual learning 
        )

        trainer.train()
        

    def end_task(self):
        pass
