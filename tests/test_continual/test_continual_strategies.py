def test_stfpm_fine_tuning():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
    from moviad.scenarios.continual.continual_trainer import ContinualTrainer
    from moviad.scenarios.continual.continual_dataset import ContinualDataset
    from moviad.scenarios.continual.strategies.fine_tuning import FineTuning
    from moviad.datasets.mvtec import MVTecDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)
    model = STFPM(teacher, student).to(device)

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }

    continual_dataset = ContinualDataset(
        DatasetArguments(**args),
        MVTecDataset
    )

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
        logger=None
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))

def test_stfpm_replay():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
    from moviad.scenarios.continual.continual_trainer import ContinualTrainer
    from moviad.scenarios.continual.continual_dataset import ContinualDataset
    from moviad.scenarios.continual.strategies.replay.replay_model import Replay
    from moviad.datasets.mvtec import MVTecDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)
    model = STFPM(teacher, student).to(device)

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }

    continual_dataset = ContinualDataset(
        DatasetArguments(**args),
        MVTecDataset
    )

    continual_model = Replay(model, 100, 0.5)

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
        logger=None
    )

    # check for parameter updates
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))