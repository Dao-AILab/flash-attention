import pytest
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func
from test_flash_attn import attention_ref

def create_mask(seqlen, lens):
    mask = torch.zeros((len(lens), seqlen), dtype=torch.bool, device="cuda")
    for i in range(len(lens)):
        mask[i, : lens[i]] = 1
    return mask

def create_pos_id(lens, ids=[0, 1, 2]):
    pos_ids = []
    for i in range(len(lens)):
        for l, id in zip(lens[i], ids):
            pos_ids.append(torch.ones((l, ), dtype=torch.int, device="cuda") * id)
    return torch.cat(pos_ids)

batch_size = 3
causal = True

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("heads", [32, 16])
@pytest.mark.parametrize(
    "random_range",
    [
        (20, 70),
        (128, 217),
        (10, 211),
        (108, 256),
        (256, 512),
    ],
)
def test_flash_attn_tree_attention_output(heads, mha_type, head_dim, dtype, random_range):
    if mha_type == "mha":
        kv_heads = heads
    elif mha_type == "gqa":
        kv_heads = 1
    else:
        kv_heads = 8

    prompt_lens = torch.randint(*random_range, (batch_size, ), device="cuda").repeat_interleave(2)
    half_prompts_lens = prompt_lens.clone()
    half_prompts_lens[0::2].zero_()
    tot_lens = torch.randint(*random_range, (batch_size * 2, ), device="cuda") + prompt_lens
    max_seqlen = tot_lens.max().item()

    mask_ref = create_mask(max_seqlen, tot_lens)
    mask = mask_ref & ~create_mask(max_seqlen, half_prompts_lens)

    q = torch.randn((batch_size * 2, max_seqlen, heads, head_dim), dtype=dtype, device="cuda")
    k = torch.randn((batch_size * 2, max_seqlen, kv_heads, head_dim), dtype=dtype, device="cuda")
    v = torch.randn((batch_size * 2, max_seqlen, kv_heads, head_dim), dtype=dtype, device="cuda")

    for i in range(0, len(prompt_lens), 2):
        q[i, :prompt_lens[i]] = q[i+1, :prompt_lens[i]] 
        k[i, :prompt_lens[i]] = k[i+1, :prompt_lens[i]] 
        v[i, :prompt_lens[i]] = v[i+1, :prompt_lens[i]] 

    lens = torch.column_stack([prompt_lens[::2], tot_lens[::2] - prompt_lens[::2], tot_lens[1::2] - prompt_lens[::2]])
    seqlens = lens.sum(1)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0)).int()
    max_seqlen_varlen = seqlens.max().item()

    start_pos_id = create_pos_id(lens, [0, 1, 2])
    end_pos_id = create_pos_id(lens, [2, 1, 2])

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    output, _ = attention_ref(q, k, v, mask_ref, mask_ref, causal=True)
    grad_output = torch.rand_like(output)
    grad_output.masked_fill_(~mask[..., None, None], 0)
    output.backward(grad_output)

    q_f = q.detach().clone().requires_grad_(True)
    k_f = k.detach().clone().requires_grad_(True)
    v_f = v.detach().clone().requires_grad_(True)
    output_f = flash_attn_func(q_f, k_f, v_f, causal=True)
    output_f.backward(grad_output)

    q_pack = q.detach().clone()[mask].requires_grad_(True)
    k_pack = k.detach().clone()[mask].requires_grad_(True)
    v_pack = v.detach().clone()[mask].requires_grad_(True)
    output_pack = flash_attn_varlen_func(q_pack, k_pack, v_pack, 
                                        cu_seqlens, cu_seqlens, 
                                        max_seqlen_varlen, max_seqlen_varlen, 
                                        causal=True, 
                                        tree_end_position_id_k=end_pos_id,
                                        tree_start_position_id_q=start_pos_id)
    output_pack.backward(grad_output[mask])

    def merge(x):
        x = x.clone().detach()
        for i in range(0, len(prompt_lens), 2):
            x[i, :prompt_lens[i]] += x[i+1, :prompt_lens[i]]
        return x

    def ck(name, x, y, z):
        x = x.detach().float()
        y = y.detach().float()
        z = z.detach().float()
        print(f"{name} max diff: {(x - y).abs().max().item()}")
        print(f"{name} mean diff: {(x - y).abs().mean().item()}")

        print(f"{name} ref max diff: {(x - z).abs().max().item()}")
        print(f"{name} ref mean diff: {(x - z).abs().mean().item()}")

        assert (x - y).abs().max().item() <= 2 * (x - z).abs().max().item()

    ck("Output", output[mask], output_pack, output_f[mask])
    ck("dQ", merge(q.grad)[mask], q_pack.grad, merge(q_f.grad)[mask])
    ck("dK", merge(k.grad)[mask], k_pack.grad, merge(k_f.grad)[mask])
    ck("dV", merge(v.grad)[mask], v_pack.grad, merge(v_f.grad)[mask])