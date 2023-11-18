from dataclasses import dataclass
from abc import ABC
from enum import Enum

class MC_TASKS(str, Enum):
    MMLU = "mc_mmlu"
    HELLA_SWAG = "mc_hellaswag"
    ARC = "mc_arc"
    TRUTHFULQA = "mc_truthfulqa"

class GEN_TASKS(str, Enum):
    LAMBADA = "gen_lambada"
    TRUTHFULQA = "gen_truthfulqa"

class KOR_TASKS(str, Enum):
    QUICK_KOR = "quick_kor"
    FULL_KOR = "full_kor"

@dataclass
class EVAL_STORED:
    task: str
    started: str = ""
    ended: str = ""
    eval: dict = None

class EVAL_BASE(ABC):
    pass

@dataclass
class EVAL_IN_PROGRESS(EVAL_BASE):
    task: str = ""
    message: str = ""
    # Timestamp
    started: str = ""

@dataclass
class EVAL_SUCCESS(EVAL_BASE):
    task: str
    message: str = ""
    # Timestamp
    started: str = ""
    ended: str = ""
    eval: dict = None

@dataclass
class EVAL_FAILURE(EVAL_BASE):
    task: str
    message: str = ""
    # Timestamp
    started: str = ""
    ended: str = ""