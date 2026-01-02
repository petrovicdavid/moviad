from typing import List, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm import tqdm

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs
from moviad.models.stfpm.loss_functions import stfpm_loss

@dataclass
class STFPMTrainArgs(TrainingArgs):

    def init_train(self, model: VADModel):
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=0.4, weight_decay=1e-4, momentum=0.9
            )
        if self.loss_function is None:
            self.loss_function = stfpm_loss


class STFPM(VADModel):
    def __init__(
        self,
        teacher: CustomFeatureExtractor,
        student: CustomFeatureExtractor,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, batch: torch.Tensor):
        if self.training:
            teacher_features, student_features = None, None
            with torch.no_grad():
                teacher_features = self.teacher(batch)
            student_features = self.student(batch)

            return teacher_features, student_features

        else:
            student_features = self.student(batch)
            teacher_features = self.teacher(batch)

            return self.post_process(
                teacher_features, student_features, batch.shape[2:]
            )

    def __call__(self, batch: torch.Tensor):
        return self.forward(batch)

    def train(self, mode: bool = True):
        self.teacher.model.eval()
        self.student.model.train(mode)
        return super().train(mode)

    def eval(self, *args, **kwargs):
        self.teacher.model.eval()
        self.student.model.eval()
        return super().eval(*args, **kwargs)

    def parameters(self):
        return self.student.model.parameters()

    def train_epoch(
        self, epoch, train_dataloader, device, training_args: STFPMTrainArgs
    ):
        loss_function = training_args.loss_function

        avg_batch_loss = 0

        # train the model
        for batch in tqdm(train_dataloader):
            batch = batch.to(device)
            teacher_features, student_features = self.forward(batch)

            for i in range(len(student_features)):
                teacher_features[i] = F.normalize(teacher_features[i], dim=1)
                student_features[i] = F.normalize(student_features[i], dim=1)
                loss = loss_function(teacher_features[i], student_features[i])

            avg_batch_loss += loss.item()

            training_args.optimizer.zero_grad()
            loss.backward()
            training_args.optimizer.step()

        avg_batch_loss /= len(train_dataloader)
        return avg_batch_loss

    def post_process(self, t_feat, s_feat, output_shape) -> torch.Tensor:
        """
        This method actually produces the anomaly maps for evalution purposes

        Args:
            - t_feat: teacher features maps
            - s_feat: student features maps

        Returns:
            - anomaly maps

        """

        device = t_feat[0].device
        score_maps = torch.tensor([1.0], device=device)
        for j in range(len(t_feat)):
            t_feat[j] = F.normalize(t_feat[j], dim=1)
            s_feat[j] = F.normalize(s_feat[j], dim=1)
            sm = torch.sum((t_feat[j] - s_feat[j]) ** 2, 1, keepdim=True)
            sm = F.interpolate(
                sm, size=output_shape, mode="bilinear", align_corners=False
            )
            # aggregate score map by element-wise product
            score_maps = score_maps * sm

        anomaly_scores = torch.max(score_maps.view(score_maps.size(0), -1), dim=1)[0]
        return score_maps, anomaly_scores
