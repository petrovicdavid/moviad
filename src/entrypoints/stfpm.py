import os
from dataclasses import dataclass, field
from glob import glob
from typing import Optional, List, Any

import torch
from torch.utils.data import DataLoader

from common.args import Args
from models.stfpm.stfpm import STFPM
from datasets.iad_dataset import IadDataset
from entrypoints.common import load_datasets
from trainers.trainer_paste import train_param_grid_search
from utilities.evaluation.evaluator import Evaluator


@dataclass
class STFPMArgs(Args):
    train_dataset: IadDataset = None
    test_dataset: IadDataset = None
    epochs = [10]
    categories: list[str] = None
    backbone: str = None
    save_path: str = None  # Default save path (no path by default)
    model_checkpoint_path: str = None  # Path for loading a model checkpoint
    visual_test_path: str = None  # Path for saving visual test outputs
    ad_layers: List[int | str] = field(default_factory=list)
    img_input_size: tuple[int, int] = (224, 224)
    img_output_size: tuple[int, int] = (224, 224)
    early_stopping: float = False
    checkpoint_dir = "./checkpoints"
    log_dirpath = "./logs"
    normalize_dataset: bool = True
    student_bootstrap_layer = [0]
    seeds = [3]
    results_dirpath: str = "./results"
    input_sizes: dict = field(
        default_factory=lambda: {
            "mcunet-in3": (176, 176),
            "mobilenet_v2": (224, 224),
            "phinet_1.2_0.5_6_downsampling": (224, 224),
            "wide_resnet50_2": (224, 224),
            "micronet-m1": (224, 224),
        }
    )
    ad_model: str = None
    trained_models_filepaths: Optional[List[str]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_dataset": self.train_dataset,
            "test_dataset": self.test_dataset,
            "epochs": self.epochs,
            "categories": self.categories,
            "backbone": self.backbone,
            "backbone_model_name": self.backbone,
            "contamination_ratio": self.contamination_ratio,
            "batch_size": self.batch_size,
            "device": self.device,
            "save_path": self.save_path,
            "model_checkpoint_path": self.model_checkpoint_path,
            "visual_test_path": self.visual_test_path,
            "ad_layers": self.ad_layers,
            "img_input_size": self.img_input_size,
            "img_output_size": self.img_output_size,
            "early_stopping": self.early_stopping,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dirpath": self.log_dirpath,
            "normalize_dataset": self.normalize_dataset,
            "student_bootstrap_layer": self.student_bootstrap_layer,
            "seeds": self.seeds,
            "results_dirpath": self.results_dirpath,
            "input_sizes": self.input_sizes,
            "trained_models_filepaths": self.trained_models_filepaths,
            "ad_model": self.ad_model,
        }


def train_stfpm(params: STFPMArgs, logger=None, evaluate=True) -> None:
    print(f"Training with params: {params}")
    train_dataset, test_dataset = load_datasets(
        params.dataset_config, params.dataset_type, params.categories[0]
    )
    params.train_dataset = train_dataset
    params.test_dataset = test_dataset
    params.epochs = params.epochs * len(params.ad_layers)
    trained_models_filepaths = train_param_grid_search(params.to_dict(), logger)
    if evaluate:
        test_stfpm(params, logger)
    m = "\n".join(trained_models_filepaths)
    print(f"Trained models:{m}")


def test_stfpm(params: STFPMArgs, logger=None) -> None:
    if params.trained_models_filepaths is None:
        params.trained_models_filepaths = glob(
            os.path.join(params.checkpoint_dir, "**/*.pth.tar"), recursive=True
        )
    if len(params.trained_models_filepaths) == 0:
        raise ValueError(f"No trained models found in {params.checkpoint_dir}")

    if not os.path.exists(params.results_dirpath):
        os.makedirs(params.results_dirpath)

    models = list(params.input_sizes.keys())

    for checkpoint_path in params.trained_models_filepaths:
        torch.manual_seed(0)

        # get category from dirname
        category = os.path.basename(os.path.dirname(checkpoint_path))

        # get backbone model name from filename
        backbone_model_name = [
            m for m in models if m in os.path.basename(checkpoint_path)
        ][0]
        img_input_size = params.input_sizes[backbone_model_name]

        print(f"backbone model name: {backbone_model_name}")
        print(f"img_input_size: {img_input_size}")
        print(f"Category: {category}")

        test_dataloader = DataLoader(
            params.test_dataset, batch_size=params.batch_size, shuffle=True
        )
        print(f"Length test dataset: {len(params.test_dataset)}")

        # load the model snapshot

        model = STFPM(
            input_size=params.img_input_size, output_size=params.img_output_size
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=params.device), strict=False
        )

        if logger is not None:
            logger.watch(model)

        model.to(params.device)

        # evaluate the model
        evaluator = Evaluator(dataloader=test_dataloader, device=params.device)
        scores = evaluator.evaluate(model, logger)

        # save the scores
        metrics_filename = os.path.join(
            params.results_dirpath,
            f"metrics_{backbone_model_name}.csv",
        )
        append_results(
            metrics_filename,
            category,
            model.seed,
            *scores,
            params.ad_model,
            str(model.ad_layers),
            backbone_model_name,
            model.weights_name,
            model.student_bootstrap_layer,
            model.epochs,
            img_input_size,
            params.img_output_size,
        )

        torch.cuda.empty_cache()
