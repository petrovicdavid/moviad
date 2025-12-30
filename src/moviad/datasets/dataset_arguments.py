
from dataclasses import dataclass
from typing import Optional
from moviad.utilities.configurations import Split

@dataclass
class DatasetArguments:
    dataset_path: str
    category: str
    split: Split | list[Split]
    img_size: Optional[tuple] = None
    gt_mask_size: Optional[tuple] = None
    image_transform_list: Optional[list] = None
