import torch

from transformers.utils import WEIGHTS_NAME
from transformers.utils.hub import cached_file


def state_dict_from_pretrained(model_name):
    return torch.load(cached_file(model_name, WEIGHTS_NAME))
