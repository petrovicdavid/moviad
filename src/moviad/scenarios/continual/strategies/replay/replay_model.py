import torch
from tqdm import trange

from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel
from moviad.trainers.trainer import Trainer
from moviad.scenarios.continual.strategies.replay.replay_memory import Memory
from moviad.utilities.evaluation.evaluator import Evaluator

class Replay(ContinualModel):

    def __init__(self, vad_model: VADModel, memory_size: int = 100, replay_ratio=0.5):
        super().__init__(vad_model)
        self.memory = Memory(memory_size=memory_size)
        self.replay_ratio = replay_ratio

    def start_task(self):
        pass

    def train_task(self, task_index: int, train_dataset, eval_dataset, train_args, metrics, device, logger):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4   
        )
        
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=train_args.batch_size,
            shuffle=False,
            num_workers=4
        ) if eval_dataset is not None else None

        train_args.init_train(self.model)
        best_metrics = {metric.name: 0.0 for metric in self.metrics}
        n_replay_samples = int(train_args.batch_size * self.replay_ratio)

        for epoch in range(train_args.epochs):

            self.vad_model.train()

            print(f"EPOCH: {epoch}")

            avg_batch_loss = 0.0

            for batch in train_dataloader:

                # combine with samples from memory
                if self.memory.num_tasks > 0:
                    memory_samples = self.memory.get_samples(n_replay_samples)
                    batch = torch.cat((batch, memory_samples), dim=0)

                avg_batch_loss += self.vad_model.train_step(batch, device, train_args)

                # update memory with new samples
                self.memory.add_samples(batch)

            avg_batch_loss /= len(train_dataloader)

            if self.logger:
                self.logger.log({
                    f"Task_T{task_index}/train/epoch": epoch,
                    f"Task_T{task_index}/train/train_loss": avg_batch_loss,
                })

            if (epoch + 1) % self.train_args.evaluation_epoch_interval == 0:
                print("Evaluating model...")
                results = Evaluator.evaluate(self.vad_model, eval_dataloader, self.metrics, self.device)

                # save the model if needed
                self.save_model(best_metrics, results)

                # update the best metrics
                best_metrics = Trainer.update_best_metrics(best_metrics, results)

                print("Training performances:")
                Trainer.print_metrics(results)

                if self.logger is not None:
                    if self.logging_prefix is not None:
                        self.logger.log({
                            f"{self.logging_prefix}train/{metric_name}": value for metric_name, value in results.items()
                        })
        

    def end_task(self):
        pass
