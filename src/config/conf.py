from dataclasses import dataclass
from omegaconf import MISSING

from .model_conf import GenModelParams
from .data_conf import DataSplitParams, DatasetParams
from .general_conf import GeneralParams


@dataclass
class Config:
    model: GenModelParams = MISSING
    data: DatasetParams = MISSING
    data_work: DataSplitParams = MISSING
    general: GeneralParams = MISSING