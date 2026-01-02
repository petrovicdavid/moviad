import torch

from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import Metric

class ContinualTrainer:

    def __init__(self, continual_dataset: ContinualDataset, model: ContinualModel, device, metrics: list[Metric], training_args: TrainingArgs):
        """
        Args:
            continual_dataset (ContinualDataset): continual dataset to be used for training
            model (nn.Module): model to be trained
            trainer_arguments (TrainerArguments): arguments for the trainer
        """
        self.continual_dataset = continual_dataset
        self.model = model
        self.trainer_arguments = training_args
        self.metrics = metrics
        self.device = device

    
    def train(self):

        for task_index in range(len(self.continual_dataset)):
            train_dataset, test_dataset = self.continual_dataset.get_task_data(task_index)

            self.model.start_task()

            self.model.train_task(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                metrics=self.metrics,
                train_args=self.trainer_arguments,
                device=self.device,
            )

            self.model.end_task()

            previous_tasks_index = self.continual_dataset.get_previous_tasks(task_index)

            for prev_task_index in previous_tasks_index:
                eval_dataset = self.continual_dataset.get_task_data_evaluation(prev_task_index)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=self.trainer_arguments.batch_size,
                    shuffle=False,
                    num_workers=4
                )
                Evaluator.evaluate(self.model, eval_dataloader, self.metrics, device=self.device)
