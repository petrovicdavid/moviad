def test_model_create_train():
    from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
    from moviad.models.stfpm.stfpm import STFPM, STFPMTrainArgs
    from moviad.trainers.trainer import Trainer
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.utilities.evaluation.metrics import RocAuc
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=True)    
    student = CustomFeatureExtractor("wide_resnet50_2", ["layer1", "layer2", "layer3"], device, frozen=False)

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "category" : "bottle",
        "split" : "train",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }
    train_dataset = MVTecDataset(DatasetArguments(**args))
    train_dataset.load_dataset()

    args["split"] = "test"
    test_dataset = MVTecDataset(DatasetArguments(**args))
    test_dataset.load_dataset()

    model = STFPM(teacher, student).to(device)
    training_args = STFPMTrainArgs(epochs=1, batch_size=4)
    training_args.init_train(model)

    trainer = Trainer(
        training_args,
        model,
        train_dataset,
        test_dataset,
        metrics=[RocAuc],
        device=device,
        logger=None,
        save_path=None,
        saving_criteria=None,
    )

    print(trainer.train_dataloader)

    # assert minimo: almeno un parametro è aggiornato
    params_before = [p.clone() for p in model.student.model.parameters()]
    trainer.train()
    params_after = [p for p in model.student.model.parameters()]
    assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))


                                                     