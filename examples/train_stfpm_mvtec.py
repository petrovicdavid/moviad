# === IMPORTS ===
import torch
from torch.utils.data import DataLoader
# Model imports
from moviad.models.stfpm.stfpm import STFPM
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.trainers.trainer_stfpm import TrainerSTFPM
# Dataset imports
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split
# Evaluation imports
from moviad.utilities.evaluation.evaluator import Evaluator
from moviad.utilities.evaluation.metrics import MetricLvl, RocAuc, F1, ProAuc

# === CONFIG ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Dataset parameters
DATASET_ROOT = "disk/datasets/mvtec"
CATEGORY = "bottle"
# Model parameters
IMG_SIZE = (224, 224)
BACKBONE = "resnet18"
LAYERS = ["layer1", "layer2", "layer3"]
# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
EVAL_INTERVAL = 10
SAVE_PATH = f"disk/checkpoints/stfpm_{BACKBONE}_{CATEGORY}_best.pth"

# === MODEL ===
teacher = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=True)
student = CustomFeatureExtractor(BACKBONE, LAYERS, device, frozen=False)
model = STFPM(teacher, student).to(device)

# === DATASET ===
ds_args = {
    "task": TaskType.SEGMENTATION,
    "root": DATASET_ROOT,
    "category": CATEGORY,
    "img_size": IMG_SIZE,
}
# train loader
train_dataset = MVTecDataset(**ds_args, split=Split.TRAIN)
train_dataset.load_dataset()
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# validation loader
val_dataset = MVTecDataset(**ds_args, split=Split.VALID)
val_dataset.load_dataset()
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test loader
test_dataset = MVTecDataset(**ds_args, split=Split.TEST)
test_dataset.load_dataset()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === METRICS ===
val_metrics = [RocAuc(MetricLvl.IMAGE), F1(MetricLvl.PIXEL)]
test_metrics = [
    RocAuc(MetricLvl.IMAGE),
    RocAuc(MetricLvl.PIXEL),
    F1(MetricLvl.PIXEL),
    ProAuc(MetricLvl.PIXEL),
]

# === TRAINING ===
trainer = TrainerSTFPM(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=test_loader,
    device=device,
    logger=None,
    save_path=SAVE_PATH,
    saving_criteria=None,
)
trainer.train(EPOCHS, EVAL_INTERVAL)

# === TESTING ===
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()
evaluator = Evaluator(test_loader, test_metrics, device)
test_results = evaluator.evaluate(model)
print(test_results)

