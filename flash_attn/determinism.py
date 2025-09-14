from dataclasses import dataclass
import os

@dataclass
class _DeterministicState:
    enabled: bool = False
    split_size: int | None = None
    reduction_tree: str = "pairwise"

_state = _DeterministicState()

def set_deterministic_mode(enabled, split_size=None, reduction_tree="pairwise"):
    global _state
    _state = _DeterministicState(enabled, split_size, reduction_tree)

    os.environ["FA2_DETERMINISTIC"] = "1" if enabled else "0"
    if split_size is not None:
        os.environ["FA2_SPLIT_SIZE"] = str(int(split_size)) 
    os.environ["FA2_REDUCTION_TREE"] = reduction_tree

def get_deterministic_mode():
    return _state