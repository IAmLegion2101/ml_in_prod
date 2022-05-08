from typing import Any
from dataclasses import dataclass
from omegaconf import MISSING


@dataclass
class GeneralParams:
    random_seed: int = MISSING
    reports_path: str = MISSING
    models_path: str = MISSING
    score_func: Any = MISSING