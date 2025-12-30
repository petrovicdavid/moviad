from __future__ import annotations

from typing import Callable
from tqdm import tqdm
import torch
import numpy as np

from .metrics import MetricLvl, Metric


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())


def append(prev, new, dtype=None, to_numpy=True):
    new = new.cpu().numpy() if to_numpy else new
    new = new.astype(dtype) if dtype else new
    return np.concatenate((prev, new), axis=0)


class Evaluator:
    """
    This class will evaluate the trained model on the test set
    and it will produce the evaluation metrics needed

    Args:
        test_dataloader (Dataloader): test dataloader
        device (torch.device): device where to run the model
    """


    def evaluate(self, model, dataloader, metrics: list[Metric], device, postprocess: Callable = min_max_norm) -> dict:
        """
        Args:
            model: a model object on which you can call model.predict(batched_images)
                and returns a tuple of anomaly_maps and anomaly_scores
            postprocess (Callable): a function to postprocess the predicted anomaly maps
        Returns:
            dict: a dictionary with metric names as keys and computed metric values as values
        """
        model.eval()

        # Initialize results as numpy arrays
        def init(*t):
            return tuple(np.empty((0,), dtype=t_) for t_ in t)
        gt_mask, gt_label, pred_anom_map, pred_anom_score = init(int, *(float,) * 3)

        for image, label, mask, path in tqdm(self.dataloader, desc="Eval"):
            with torch.no_grad():  # get anomaly map and score
                anom_maps, anom_scores = model(image.to(self.device))

            # Append ground truth, anomaly scores, and predicted masks
            gt_mask = append(gt_mask, mask, dtype=int)
            gt_label = append(gt_label, label)
            pred_anom_map = append(pred_anom_map, anom_maps)
            pred_anom_score = append(pred_anom_score, anom_scores)

        pred_anom_map = postprocess(pred_anom_map)

        report = {}
        for metric in self.metrics:
            if metric.level == MetricLvl.IMAGE:
                pred = pred_anom_score
                gt = gt_label
            elif metric.level == MetricLvl.PIXEL:
                pred = pred_anom_map
                gt = gt_mask
            report[metric.name] = metric.compute(gt, pred)

        return report

