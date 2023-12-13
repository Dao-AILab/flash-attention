import torch
from torch.nn.functional import scaled_dot_product_attention
from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
from flash_attn.flash_attn_interface import get_kvcache_block_size, flash_attn_with_blocked_kvcache


b, s, h_q, h_kv, d = 1, 131072, 32, 1, 512
block_size = get_kvcache_block_size(d)
dtype = torch.bfloat16
device = torch.device("cuda:5")
torch.set_default_dtype(dtype)
torch.set_default_device(device)
torch.cuda.set_device(device)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item()


def timer(func):
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    e = 100
    for _ in range(e):
        func()
    en.record()
    torch.cuda.synchronize()
    t = st.elapsed_time(en) / e
    bytes = b * s * h_kv * d * 2 * (torch.finfo(dtype).bits // 8)

    print(f"{t} ms, {bytes / 10**6 / t} GB/s")
    return t


@torch.inference_mode()
def test_flash_attention():
    q = torch.randn(b, 1, h_q, d)
    k = torch.randn(b, s, h_kv, d)
    v = torch.randn(b, s, h_kv, d)
    blocked_k = k.view(-1, block_size, h_kv, d)
    blocked_v = v.view(-1, block_size, h_kv, d)
    block_table = torch.arange(b * s // block_size, dtype=torch.int32).view(b, s // block_size)
    cache_seqlens = torch.full((b,), s, dtype=torch.int32)

    # Warm up
    for _ in range(100):
        torch.ones(1 << 20)

    timer(lambda: flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens))
    timer(lambda: scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
    timer(lambda: flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens))
    timer(lambda: scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
    timer(lambda: flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens))
    timer(lambda: scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))

    out_block = flash_attn_with_blocked_kvcache(q, blocked_k, blocked_v, block_table, cache_seqlens)
    out_flash = scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    print(calc_diff(out_block, out_flash))
    torch.testing.assert_close(out_block, out_flash)


if __name__ == "__main__":
    test_flash_attention()
