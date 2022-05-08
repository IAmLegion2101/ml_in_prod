from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

@dataclass
class GenModelParams:
    _target_: str = MISSING

@dataclass
class GaussNBParams(GenModelParams):
    var_smoothing: float = MISSING
