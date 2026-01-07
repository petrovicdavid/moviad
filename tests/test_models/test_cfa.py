def test_model_create_train():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.cfa import CFA, CFATrainArgs
    from moviad.trainers.trainer import Trainer
    from moviad.datasets.mvtec import MVTecDataset
    from torch.utils.data import Subset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, AvgPrec, F1, ProAuc
    import torch
    import wandb

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="moviad_test", name="cfa", mode="disabled")

    feature_extractor = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], frozen=True)    

    args = DatasetArguments(
        dataset_path = "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        img_size = (256, 256),
        gt_mask_size = (256, 256),
        image_transform_list = None
    )

    train_dataset = MVTecDataset(args, category="bottle", split="train")
    train_dataset = Subset(train_dataset, list(range(0, 10)))  # use a subset for faster testing
    test_dataset = MVTecDataset(args, category="bottle", split="test")

    model = CFA(feature_extractor, "wide_resnet50_2")
    model.to(device)


    training_args = CFATrainArgs(epochs=2, batch_size=4)
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
    params_before = [p.clone() for p in model.parameters()]
    trainer.train()
    params_after = [p for p in model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))



                                                     