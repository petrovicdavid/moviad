from moviad.scenarios.continual.continual_model import ContinualModel
from moviad.models.patchcore.patchcore import PatchCore
from moviad.models.training_args import TrainingArgs

import torch
from tqdm import tqdm

class PatchCoreCL(ContinualModel):

    def __init__(self, patchcore_model: PatchCore):
        super().__init__(patchcore_model)
        self.vad_model.memory_bank = {}

    def _rebalance_memory_bank(self):
        n_tasks = len(self.vad_model.memory_bank) + 1

        for task_id in self.vad_model.memory_bank:
            embeddings = self.vad_model.memory_bank[task_id]
            n_samples = embeddings.shape[0]
            target_n_samples = self.vad_model.memory_bank_size // n_tasks

            if n_samples > target_n_samples:
                self.vad_model.coreset_extractor.k = target_n_samples
                coreset = self.coreset_extractor.extract_coreset(embeddings)
                self.vad_model.memory_bank[task_id] = coreset

    def start_task(self, train_args: TrainingArgs = None):
        self._rebalance_memory_bank()

    def train_task(self, task_index: int, train_dataset, eval_dataset, metrics, device, logger = None, train_args:TrainingArgs = None):

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_args.batch_size,
            shuffle=True,
            num_workers=4   
        )
        
        embeddings = []

        with torch.no_grad():

            print("Embedding Extraction:")
            for batch in tqdm(iter(train_dataloader)):
                embedding = self(batch.to(device))
                embeddings.append(embedding)

            embeddings = torch.cat(embeddings, dim = 0)
            torch.cuda.empty_cache()

            #apply coreset reduction
            print("Coreset Extraction:")
            coreset = self.coreset_extractor.extract_coreset(embeddings)

            self.vad_model.memory_bank[task_index] = coreset
        

    def end_task():
        pass

    def forward(self, batch: torch.Tensor):
        
        anomaly_maps = []
        pred_scores = []

        for task_id in self.vad_model.memory_bank:
            anomaly_maps, pred_scores = [], []
            task_memory_bank = self.vad_model.memory_bank[task_id]

            anomaly_maps, scores = self.vad_model.calculate_anomaly_maps_scores(
                embedding=self.vad_model.feature_extractor(batch),
                memory_bank=task_memory_bank,
                batch_size=batch.shape[0],
                width=batch.shape[2],
                height=batch.shape[3]
            )

            anomaly_maps.append(anomaly_maps)
            pred_scores.append(scores)

        anomaly_maps = torch.stack(anomaly_maps, dim=0)
        anomaly_scores = torch.stack(pred_scores, dim=0)
   
        num_tasks = len(self.vad_model.memory_bank)

        anomaly_maps = anomaly_maps.view(-1, num_tasks, anomaly_maps.size(1), anomaly_maps.size(2), anomaly_maps.size(3))
        anomaly_scores = anomaly_scores.view(-1, num_tasks)             

        min_scores = anomaly_scores.argmin(dim=1)
        batch_idx = torch.arange(anomaly_scores.size(0))

        min_anomaly_maps = anomaly_maps[batch_idx, min_scores]
        min_scores = anomaly_scores[batch_idx, min_scores]

        return min_anomaly_maps, min_scores




