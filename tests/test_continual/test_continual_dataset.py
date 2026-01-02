def test_continual_dataset():
    from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    from moviad.scenarios.continual.continual_dataset import ContinualDataset

    args = {
        "dataset_path" : "/mnt/mydisk/manuel_barusco/datasets/mvtec",
        "img_size" : (256, 256),
        "gt_mask_size" : (256, 256),
        "image_transform_list" : None
    }

    continual_dataset = ContinualDataset(
        dataset_class=MVTecDataset,
        dataset_arguments=DatasetArguments(**args),
    )

    assert len(continual_dataset) == len(continual_dataset.categories)

    for task_index in range(len(continual_dataset)):
        train_dataset, test_dataset = continual_dataset.get_task_data(task_index)
        assert isinstance(train_dataset, MVTecDataset)
        assert isinstance(test_dataset, MVTecDataset)
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0