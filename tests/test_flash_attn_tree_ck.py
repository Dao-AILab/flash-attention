from numpy import dtype
import pytest
import torch
import torch.nn.functional as F
from itertools import chain
from flash_attn import flash_attn_func, flash_attn_varlen_func
from test_flash_attn import attention_ref

def get_single_grad(x):
    grad = x.grad
    x.grad = None
    return grad

def get_grad(*args):
    grads = [torch.cat(list(chain(*[[get_single_grad(sub2) for sub2 in sub] for sub in arg]))) for arg in args]
    return grads

def prepare_tree_attn(q_chunks, k_chunks, v_chunks, chunk_lens):
    q = torch.cat(list(chain(*q_chunks)))
    k = torch.cat(list(chain(*k_chunks)))
    v = torch.cat(list(chain(*v_chunks)))
    lens = torch.tensor(list(map(lambda x: sum(x), chunk_lens)), dtype=torch.int, device="cuda")
    max_seqlen = lens.max().item()
    cu_seqlens = F.pad(torch.cumsum(lens, dim=0), pad=(1, 0)).int()
    tree_dfs_order_start_q = []
    tree_dfs_order_end_k = []
    for lens in chunk_lens:
        for i in range(len(lens)):
            if i == 0:
                tree_dfs_order_end_k.extend([len(lens) - 1] * lens[0])
            else:
                tree_dfs_order_end_k.extend([i] * lens[i])
            tree_dfs_order_start_q.extend([i] * lens[i])

    tree_dfs_order_start_q = torch.tensor(tree_dfs_order_start_q, device="cuda", dtype=torch.int)
    tree_dfs_order_end_k = torch.tensor(tree_dfs_order_end_k, device="cuda", dtype=torch.int)
    return q, k, v, max_seqlen, cu_seqlens, tree_dfs_order_end_k, tree_dfs_order_start_q

def copy_prompt(chunks):
    copyed_chunks = []
    for chunk in chunks:
        for r in chunk[1:]:
            copyed_chunks.append([chunk[0], r])
    return copyed_chunks

def prepare_varlen(*args):
    args = [copy_prompt(arg) for arg in args]
    q, k, v, max_seqlen, cu_seqlens, _, _ = prepare_tree_attn(*args)
    return q, k, v, max_seqlen, cu_seqlens

def pack_varlen(out, chunk_lens):
    mask = []
    for chunk in chunk_lens:
        mask.extend([1] * (chunk[0] + chunk[1]))
        for r in chunk[2:]:
            mask.extend([0] * chunk[0] + [1] * r)
    mask = torch.tensor(mask, dtype=torch.bool, device=out.device)
    return out[mask]

def pad(chunks):
    copyed_chunks = []
    for chunk in chunks:
        for r in chunk[1:]:
            copyed_chunks.append(torch.cat([chunk[0], r]))
    max_len = max(map(len, copyed_chunks))
    inputs = torch.stack([F.pad(chunk, pad=(0, 0, 0, 0, 0, max_len - len(chunk))) for chunk in copyed_chunks])
    return inputs

def prepare(*args):
    args = [pad(arg) for arg in args]
    return args

def pack(out, chunk_lens):
    mask = []
    max_len = out.size(1)
    for chunk in chunk_lens:
        mask.append([1] * (chunk[0] + chunk[1]) + [0] * (max_len - chunk[0] - chunk[1]))
        for r in chunk[2:]:
            mask.append([0] * chunk[0] + [1] * r + [0] * (max_len - chunk[0] - r))
    mask = torch.tensor(mask, dtype=torch.bool, device=out.device)
    return out[mask]

batch_size = 7
causal = True

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("heads", [32, 16])
@pytest.mark.parametrize(
    "random_range",
    [
        (20, 70),
        (10, 211),
        (30, 552),
        (108, 270),
    ],
)
def test_flash_attn_tree_attention_output(heads, mha_type, head_dim, dtype, random_range):
    if mha_type == "mha":
        kv_heads = heads
    elif mha_type == "gqa":
        kv_heads = 1
    else:
        kv_heads = 8

    chunk_lens = []
    for i in range(batch_size):
        num_responses = torch.randint(2, 3+1, ())
        chunk_lens.append(torch.randint(*random_range, (num_responses+1,)).tolist())

    kwargs = {"dtype": dtype, "device": "cuda", "requires_grad": True}
    q_chunks = [[torch.randn((l, heads, head_dim), **kwargs) for l in lens] for lens in chunk_lens]
    k_chunks = [[torch.randn((l, kv_heads, head_dim), **kwargs) for l in lens] for lens in chunk_lens]
    v_chunks = [[torch.randn((l, kv_heads, head_dim), **kwargs) for l in lens] for lens in chunk_lens]

    q, k, v, max_seqlen, cu_seqlens, tree_dfs_order_end_k, tree_dfs_order_start_q = prepare_tree_attn(q_chunks, k_chunks, v_chunks, chunk_lens)
    out_tree = flash_attn_varlen_func(q, k, v, 
                                        cu_seqlens, cu_seqlens, 
                                        max_seqlen, max_seqlen, 
                                        causal=True, 
                                        tree_dfs_order_end_k=tree_dfs_order_end_k,
                                        tree_dfs_order_start_q=tree_dfs_order_start_q)
    grad_out = torch.randn_like(out_tree)
    out_tree.backward(grad_out)
    q_tree_grad, k_tree_grad, v_tree_grad = get_grad(q_chunks, k_chunks, v_chunks)

    q, k, v, max_seqlen, cu_seqlens = prepare_varlen(q_chunks, k_chunks, v_chunks, chunk_lens)
    out_varlen = flash_attn_varlen_func(q, k, v, 
                                        cu_seqlens, cu_seqlens, 
                                        max_seqlen, max_seqlen, 
                                        causal=True)

    out_varlen_pack = pack_varlen(out_varlen, chunk_lens)
    out_varlen_pack.backward(grad_out)
    q_varlen_grad, k_varlen_grad, v_varlen_grad = get_grad(q_chunks, k_chunks, v_chunks)

    q, k, v = prepare(q_chunks, k_chunks, v_chunks)
    out_ref, _ = attention_ref(q, k, v, causal=True)
    out_ref_pack = pack(out_ref, chunk_lens)
    out_ref_pack.backward(grad_out)
    q_grad, k_grad, v_grad = get_grad(q_chunks, k_chunks, v_chunks)

    @torch.no_grad
    def ck(name, x, y, z):
        print(f"{name} varlen max diff: {(x - y).abs().max().item()}")
        print(f"{name} varlen mean diff: {(x - y).abs().mean().item()}")

        print(f"{name} tree max diff: {(x - z).abs().max().item()}")
        print(f"{name} tree mean diff: {(x - z).abs().mean().item()}")

        assert (x - z).abs().max().item() <= 2 * (x - y).abs().max().item()

    ck("Output", out_ref_pack, out_varlen_pack, out_tree)
    ck("dQ", q_grad, q_varlen_grad, q_tree_grad)
    ck("dK", k_grad, k_varlen_grad, k_tree_grad)
    ck("dV", v_grad, v_varlen_grad, v_tree_grad)