import torch
import transformers

from transformers.utils import WEIGHTS_NAME, WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from transformers.utils import is_remote_url
from transformers.modeling_utils import load_state_dict
from transformers.utils.hub import cached_file, get_checkpoint_shard_files
import os


def state_dict_from_pretrained(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = 'cpu' if dtype not in [torch.float32, None] else device
    is_sharded = False
    resolved_archive_file = None

    weights_path = os.path.join(model_name, WEIGHTS_NAME)
    weights_index_path = os.path.join(model_name, WEIGHTS_INDEX_NAME)
    safe_weights_path = os.path.join(model_name, SAFE_WEIGHTS_NAME)
    safe_weights_index_path = os.path.join(model_name, SAFE_WEIGHTS_INDEX_NAME)

    if os.path.isfile(weights_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
    elif os.path.isfile(weights_index_path):
        resolved_archive_file = cached_file(model_name, WEIGHTS_INDEX_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file is not None:
            is_sharded = True
    elif os.path.isfile(safe_weights_path):
        resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_NAME,
                                            _raise_exceptions_for_missing_entries=False)
    elif os.path.isfile(safe_weights_index_path):
        resolved_archive_file = cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME,
                                            _raise_exceptions_for_missing_entries=False)
        if resolved_archive_file is not None:
            is_sharded = True

    if resolved_archive_file is None:
        raise EnvironmentError(f"Model name {model_name} was not found.")

    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different
        # checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            model_name, resolved_archive_file
        )
        state_dict = {}
        for sharded_file in resolved_archive_file:
            state_dict.update(torch.load(sharded_file, map_location=mapped_device))
    else:
        from safetensors.torch import load_file as safe_load_file
        # state_dict = torch.load(resolved_archive_file, map_location=device)
        state_dict = safe_load_file(resolved_archive_file, device=device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict
