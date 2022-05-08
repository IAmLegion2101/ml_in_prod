from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class DataSplitParams:
    val_size: float = MISSING
    test_size: float = MISSING
    random_state: int = MISSING
    target_name: str = MISSING

@dataclass
class DatasetParams:
    dataset_fname: str = MISSING
    dataset_path: str = MISSING