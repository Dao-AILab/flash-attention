import re
from pathlib import Path

import torch
import math
from einops import rearrange

def load_checkpoint(path, device='cpu'):
    path = Path(path).expanduser()
    is_deepspeed = False
    if path.is_dir():  # DeepSpeed checkpoint
        is_deepspeed = True
        latest_path = path / 'latest'
        if latest_path.is_file():
            with open(latest_path, 'r') as fd:
                tag = fd.read().strip()
        else:
            raise ValueError(f"Unable to find 'latest' file at {latest_path}")
        path /= f'{tag}/mp_rank_00_model_states.pt'
    state_dict = torch.load(path, map_location=device)
    if is_deepspeed:
        state_dict = state_dict['module']

        # Replace the names of some of the submodules
        def key_mapping(key):
            return re.sub(r'^module.model.', '', key)

        state_dict = {key_mapping(k): v for k, v in state_dict.items()}
    return state_dict


def blockdiag_to_dense_mlp_bert(state_dict):
    from src.ops.blockdiag_multiply import blockdiag_weight_to_dense_weight
    names = {name for name in state_dict
             if re.match('bert.encoder.layer.(\d+).(mlp.fc(1|2)|(intermediate|output).dense).weight',
                         name)}
    for name in names:
        state_dict[name] = blockdiag_weight_to_dense_weight(state_dict[name])
    return state_dict

def interpolate_pos_embedding(state_dict, out_seqlen, pos_embedding_name='model.pos_encoder.pe', interleave=False):
    orig_emb = state_dict['state_dict'][pos_embedding_name]
    assert (out_seqlen % orig_emb.shape[1]) == 0, 'out_seqlen must be a multiple of the original sequence length'
    reps = [1 for i in orig_emb.shape]
    reps[1] = out_seqlen // orig_emb.shape[1]
    
    if interleave:
        assert math.isqrt(orig_emb.shape[1]) ** 2 == orig_emb.shape[1], 'interleave only works for square lengths'
        assert math.isqrt(out_seqlen) ** 2 == out_seqlen, 'interleave only works for square lengths'
        assert math.isqrt(reps[1]) ** 2 == reps[1], 'out_seqlen / seqlen must be a perfect square'

        emb_square = rearrange(orig_emb, 'b (h w) d -> b h w d', h = math.isqrt(orig_emb.shape[1]))
        emb_square_expanded = emb_square.repeat_interleave(math.isqrt(reps[1]), axis=1).repeat_interleave(math.isqrt(reps[1]), axis=2)
        new_emb = rearrange(emb_square_expanded, 'b h w d -> b (h w) d')
        state_dict['state_dict'][pos_embedding_name] = new_emb
    else:
        state_dict['state_dict'][pos_embedding_name] = orig_emb.repeat(*reps)

    ret = remove_model_prefix(state_dict)
    # # HACK: this is a hack for block-sparse flash attention
    ret = {
        k: v
        for k, v in ret.items()
        if not k.endswith('inner_attn.layout')
    }
    return ret

def remove_model_prefix(state_dict):
    # HACK: this is a hack to get the model to load properly, get rid of 'model.' prefix
    for key in list(state_dict['state_dict'].keys()):
        if key.startswith('model.'):
            new_key = key[len('model.'):]
            state_dict['state_dict'][new_key] = state_dict['state_dict'].pop(key)

    # HACK: something is wrong with the state dict being loaded...
    return state_dict['state_dict']
