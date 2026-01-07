import torch

from moviad.scenarios.continual.strategies.replay.replay_model import Replay
from moviad.models.cfa.cfa import CFA

class CFAContinual(Replay):

    def __init__(self, model:CFA, memory_size: int = 2000, **kwargs):
        super().__init__(model=model, memory_size=memory_size, **kwargs)

    def update_memory_bank(self, task_index: int, train_dataloader: torch.utils.data.DataLoader):
        memory_bank = self.vad_model.initialize_memory_bank(train_dataloader)
        new_memory_bank = memory_bank * (task_index / (task_index + 1)) + memory_bank / (task_index + 1)
        self.vad_model.memory_bank = new_memory_bank

    
    def train_task(self, task_index: int, train_dataset, eval_dataset, train_args, metrics, device, logger):
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4   
        )

        # initialize the memory bank for CFA
        if task_index == 0:
            self.vad_model.initialize_memory_bank(train_dataloader)
        else: 
            self.update_memory_bank(task_index, train_dataloader)

        # train the patch descriptor using replay strategy
        super().train_task(
            task_index=task_index,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_args=train_args,
            metrics=metrics,
            device=device,
            logger=logger,
        )

    

