import torch

from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file


def state_dict_from_pretrained(model_name, device=None, dtype=None):
    state_dict = torch.load(cached_file(model_name, WEIGHTS_NAME), map_location=device)
    if dtype is not None:
        state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    return state_dict
