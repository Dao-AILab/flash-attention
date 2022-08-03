import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from flash_attn.flash_attention import FlashMHA
from apex.normalization import FusedLayerNorm
import numpy as np
import random as rand
import logging
import logging.handlers
from tools.check_tool import is_same_matrix
from tools.window_attention import WindowAttentionFP16

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

logger = logging.getLogger("swin-T")
logger.setLevel(logging.INFO)

rf_handler = logging.StreamHandler()
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)

if __name__ == "__main__":
    dim = 192
    num_heads = 6
    window_size = 7
    qk_scale = None
    attn_drop = 0.0
    drop = 0.0
    qkv_bias = True
    batch_size = 1

    x_windows = torch.rand(batch_size, window_size * window_size, dim, device='cuda', dtype=torch.half, requires_grad=True)
    resutlt = torch.rand(batch_size, window_size * window_size, dim, device='cuda', dtype=torch.half, requires_grad=False)

    loss = nn.L1Loss()

    wqkv_weight = np.random.uniform(-1, 1, [dim * 3, dim]).astype(np.float16)
    wqkv_bias = np.random.uniform(-1, 1, [dim * 3]).astype(np.float16)
    
    out_proj_weight = np.random.uniform(-1, 1, [dim, dim]).astype(np.float16)
    out_proj_bias = np.random.uniform(-1, 1, [dim]).astype(np.float16)

    init_para = {
        'qkv_weight' : torch.from_numpy(wqkv_weight),
        'qkv_bias' : torch.from_numpy(wqkv_bias),
        'proj_weight' : torch.from_numpy(out_proj_weight),
        'proj_bias' : torch.from_numpy(out_proj_bias)
    }

    attn_mask = None

    # win
    logger.info("windows attention")
    win_attn = WindowAttentionFP16(
        dim, window_size=to_2tuple(window_size), num_heads=num_heads,
        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        init_para = init_para).cuda()
    logger.info(f"win_attn model is \n {str(win_attn)}")

    print("para in win_attn is")
    for name, para in win_attn.named_parameters():
        print(name, para, para.shape)

    optimizer_win_attn = torch.optim.SGD(win_attn.parameters(), lr=1e-3)

    win_output = win_attn(x_windows, attn_mask)
    win_loss = loss(win_output, resutlt)
    win_loss.backward()
    torch.cuda.synchronize()

    win_attn_grad = {}
    for name, parms in win_attn.named_parameters():	
        print('\nAfter backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")

        if name == "qkv.weight":
            win_attn_grad["qkv_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "qkv.bias":
            win_attn_grad["qkv_bias_grad"] = parms.grad.cpu().detach().numpy()
        if name == "proj.weight":
            win_attn_grad["proj_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "proj.bias":
            win_attn_grad["proj_bias_grad"] = parms.grad.cpu().detach().numpy()

    # fmha
    logger.info("flash attention")
    flash_attn = FlashMHA(dim, num_heads, init_para = init_para).cuda()
    logger.info(f"flash attn model is \n {str(flash_attn)}")

    print("para in flash_attn is")
    for name, para in flash_attn.named_parameters():
        print(name, para, para.shape)


    optimizer_win_fmha = torch.optim.SGD(flash_attn.parameters(), lr=1e-3)
    # flash_attn, optimizer_win_fmha = amp.initialize(flash_attn, optimizer_win_fmha, opt_level="O2")
    flash_output, attn_weights = flash_attn(x_windows, None, None, attn_mask)
    flash_loss = loss(flash_output, resutlt)
    flash_loss.backward()
    torch.cuda.synchronize()

    flash_attn_grad = {}
    for name, parms in flash_attn.named_parameters():	
        print('\nAfter backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")

        if name == "Wqkv.weight":
            flash_attn_grad["qkv_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "Wqkv.bias":
            flash_attn_grad["qkv_bias_grad"] = parms.grad.cpu().detach().numpy()
        if name == "out_proj.weight":
            flash_attn_grad["proj_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "out_proj.bias":
            flash_attn_grad["proj_bias_grad"] = parms.grad.cpu().detach().numpy()

    # check output
    is_same_matrix(flash_output.cpu().detach().numpy(), win_output.cpu().detach().numpy(), "output")

    # check dgrad
    is_same_matrix(flash_attn_grad["qkv_weight_grad"], win_attn_grad["qkv_weight_grad"], "qkv_weight_grad")
    is_same_matrix(flash_attn_grad["qkv_bias_grad"], win_attn_grad["qkv_bias_grad"], "qkv_bias_grad")
    is_same_matrix(flash_attn_grad["proj_weight_grad"], win_attn_grad["proj_weight_grad"], "proj_weight_grad")
    is_same_matrix(flash_attn_grad["proj_bias_grad"], win_attn_grad["proj_bias_grad"], "proj_bias_grad")
