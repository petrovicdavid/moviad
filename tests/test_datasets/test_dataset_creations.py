def test_mvtec_train_creation_load_sizes():
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    
    args = DatasetArguments(
        dataset_path="/mnt/mydisk/manuel_barusco/datasets/mvtec",
        category="bottle",
        split="train",
        img_size=(256, 256),
        gt_mask_size=(256, 256),
        image_transform_list=None,
    )

    dataset = MVTecDataset(args)
    dataset.load_dataset()
    assert dataset.is_loaded()
    assert len(dataset) > 0
    assert dataset[0].shape == (3, 256, 256)

def test_mvtec_test_creation_load_sizes():
    from moviad.datasets.mvtec import MVTecDataset
    from moviad.datasets.dataset_arguments import DatasetArguments
    import numpy as np
    
    args = DatasetArguments(
        dataset_path="/mnt/mydisk/manuel_barusco/datasets/mvtec",
        category="bottle",
        split="test",
        img_size=(256, 256),
        gt_mask_size=(256, 256),
        image_transform_list=None,
    )

    dataset = MVTecDataset(args)
    dataset.load_dataset()
    assert dataset.is_loaded()
    assert len(dataset) > 0
    assert dataset[0][0].shape == (3, 256, 256)
    assert dataset[0][1] == np.int64(1)
    assert dataset[0][2].shape == (1, 256, 256)
