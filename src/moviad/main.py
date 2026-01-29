from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
from moviad.trainers.trainer import Trainer
from moviad.datasets.mvtec import MVTecDataset
from torch.utils.data import Subset
from moviad.datasets.dataset_arguments import DatasetArguments
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
from moviad.scenarios.continual.continual_trainer import ContinualTrainer
from moviad.scenarios.continual.continual_dataset import ContinualDataset
from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
import torch
import wandb

import argparse


def get_test(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", type=str, required=False, help="Es. test_model_create_train", choices=["test_model_create_train"]) 
    parser.add_argument("--model", type=str, required=False, help="Es. FT_stfpm, MT_stfpm", choices=["FT_stfpm", "MT_stfpm"]) 
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Es. wide_resnet50_2", choices=["wide_resnet50_2"])
    args = parser.parse_args()

    if args.modality is None and args.model is None:
        raise ValueError("Either --modality or --model must be specified.")
    
    return args 



def test_model_create_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="moviad_test", name="stfpm")

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)

    args = DatasetArguments(
        dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
        img_size = (256, 256),
        gt_mask_size = (256, 256),
        image_transform_list = None
    )

    train_dataset = MVTecDataset(args, category="bottle", split="train")
    train_dataset = Subset(train_dataset, list(range(0, 10)))  # use a subset for faster testing

    test_dataset = MVTecDataset(args, category="bottle", split="test")

    model = STFPM(teacher, student)
    model.to(device)
    training_args = STFPMTrainArgs(epochs=2, batch_size=4)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))





def train_stfpm_FT():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="moviad_test", name="stfpm")

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)

    args = DatasetArguments(
        dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
        img_size = (256, 256),
        gt_mask_size = (256, 256),
        image_transform_list = None
    )

    continual_dataset = ContinualDataset(
        dataset_class=MVTecDataset,
        dataset_arguments=args,
    )

    model = STFPM(teacher, student)
    model.to(device)
    continual_model = FineTuning(model)

    trainer = ContinualTrainer(
        continual_dataset,
        continual_model,
        device,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        training_args=STFPMTrainArgs(epochs=2, batch_size=4),
        logger=wandb
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


### Da vedere se usare MultiTask class o no (per ora non la uso) ###
def train_stfpm_multi_task():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="moviad_test", name="stfpm")

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)

    args = DatasetArguments(
        dataset_path = "/mnt/disk1/manuel_barusco/datasets/mvtec",
        img_size = (256, 256),
        gt_mask_size = (256, 256),
        image_transform_list = None
    )

    train_dataset, test_dataset = ContinualDataset(
        dataset_class=MVTecDataset,
        dataset_arguments=args,
    ).get_all_tasks_data()


    model = STFPM(teacher, student)
    model.to(device)

    training_args = STFPMTrainArgs(epochs=10, batch_size=8)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[
            RocAuc(MetricLvl.IMAGE),
            RocAuc(MetricLvl.PIXEL),
            AvgPrec(MetricLvl.IMAGE),
            AvgPrec(MetricLvl.PIXEL),
            F1(MetricLvl.IMAGE),
            F1(MetricLvl.PIXEL),
            ProAuc(MetricLvl.PIXEL),
        ],
        device=device,
        logger=wandb,
        save_path=None,
        saving_criteria=None,
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



def main():
    args = get_test()
    if args.modality == "test_model_create_train":
        test_model_create_train()
    elif args.model == "FT_stfpm":
        train_stfpm_FT()
    elif args.model == "MT_stfpm":
        train_stfpm_multi_task()
    else:
        raise NotImplementedError(f"Model {args.model} not implemented in tests.")



if __name__ == "__main__":
    main()