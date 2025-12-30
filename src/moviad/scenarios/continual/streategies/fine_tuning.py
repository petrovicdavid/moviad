from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer

from tqdm import trange

class FineTuning(ContinualModel):

    def __init__(self, model: VADModel):
        super().__init__(model)

    def start_task(self, train_args):
        pass

    def train_task(self, train_dataset, test_dataset, metrics, train_args: TrainingArgs, device, logger):

        trainer = Trainer(
            train_args,
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
