"""Narrowing test: separate Is_even_MN from causal masking."""
import torch

def ref_bwd(q, k, v, dout, softmax_scale, causal=True):
    q_f, k_f, v_f, do_f = q.float(), k.float(), v.float(), dout.float()
    S = q_f @ k_f.T * softmax_scale
    seqlen = q.shape[0]
    if causal:
        mask = torch.triu(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool), diagonal=1)
        S.masked_fill_(mask, float('-inf'))
    P = torch.softmax(S, dim=-1)
    O = P @ v_f
    dV = P.T @ do_f
    dP = do_f @ v_f.T
    dP_sum = (do_f * O).sum(dim=-1)
    dS = P * (dP - dP_sum.unsqueeze(-1))
    dQ = dS @ k_f * softmax_scale
    dK = dS.T @ q_f * softmax_scale
    return dQ, dK, dV

def run_test(seqlen, causal, hdim=32):
    device = 'cuda'
    dtype = torch.float16
    softmax_scale = 1.0 / (hdim ** 0.5)
    torch.manual_seed(42)
    q = torch.randn(1, seqlen, 1, hdim, device=device, dtype=dtype)
    k = torch.randn(1, seqlen, 1, hdim, device=device, dtype=dtype)
    v = torch.randn(1, seqlen, 1, hdim, device=device, dtype=dtype)
    dout = torch.randn(1, seqlen, 1, hdim, device=device, dtype=dtype)

    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
    out, lse, _, _ = _flash_attn_forward(q, k, v, 0.0, softmax_scale, causal, -1, -1, 0.0, None, False)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    _flash_attn_backward(dout, q, k, v, out, lse, dq, dk, dv, 0.0, softmax_scale, causal, -1, -1, 0.0, None, False, None)

    dQ_ref, dK_ref, dV_ref = ref_bwd(q[0,:,0,:], k[0,:,0,:], v[0,:,0,:], dout[0,:,0,:], softmax_scale, causal)
    dV_err = (dv[0,:,0,:].float() - dV_ref).abs().max().item()
    dQ_err = (dq[0,:,0,:].float() - dQ_ref).abs().max().item()
    dK_err = (dk[0,:,0,:].float() - dK_ref).abs().max().item()
    tag = "causal" if causal else "non-causal"
    even = "even" if seqlen % 128 == 0 else "uneven"
    print(f"  {tag:10s} seqlen={seqlen:4d} ({even:6s}): dV={dV_err:.6f}  dQ={dQ_err:.6f}  dK={dK_err:.6f}")

if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    # Test non-causal with various seqlens (Is_even_MN varies)
    print("Non-causal:")
    for sl in [8, 16, 32, 64, 128, 256]:
        run_test(sl, causal=False)
    print("Causal:")
    for sl in [8, 16, 32, 64, 128, 256]:
        run_test(sl, causal=True)
