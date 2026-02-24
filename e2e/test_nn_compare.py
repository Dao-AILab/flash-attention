"""Compare a simple Transformer model using flash_attn vs PyTorch SDPA with actual loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


class SimpleTransformer(nn.Module):
    """Minimal Transformer: N layers of Attention + FFN, then a classification head."""
    def __init__(self, d_model, nheads, nlayers, vocab_size, use_flash=True):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        self.nheads = nheads
        self.head_dim = d_model // nheads
        self.use_flash = use_flash
        for _ in range(nlayers):
            self.layers.append(nn.ModuleDict({
                'ln1': nn.LayerNorm(d_model),
                'qkv': nn.Linear(d_model, 3 * d_model, bias=False),
                'out': nn.Linear(d_model, d_model, bias=False),
                'ln2': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, 4 * d_model, bias=False),
                'ff2': nn.Linear(4 * d_model, d_model, bias=False),
            }))
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, causal=False):
        B, S = tokens.shape
        x = self.embed(tokens)
        for layer in self.layers:
            h = layer['ln1'](x)
            qkv = layer['qkv'](h).reshape(B, S, 3, self.nheads, self.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if self.use_flash:
                attn_out = flash_attn_func(q, k, v, causal=causal)
            else:
                q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
                attn_out = attn_out.transpose(1, 2)
            attn_out = attn_out.reshape(B, S, -1)
            x = x + layer['out'](attn_out)
            x = x + layer['ff2'](F.gelu(layer['ff1'](layer['ln2'](x))))
        return self.head(x)


def compare(batch, seqlen, d_model, nheads, causal, nlayers=2, vocab_size=256):
    device = 'cuda'
    dtype = torch.float16

    torch.manual_seed(123)
    model_flash = SimpleTransformer(d_model, nheads, nlayers, vocab_size, use_flash=True).to(device=device, dtype=dtype)
    model_sdpa = SimpleTransformer(d_model, nheads, nlayers, vocab_size, use_flash=False).to(device=device, dtype=dtype)
    model_sdpa.load_state_dict(model_flash.state_dict())

    # Input tokens and target
    torch.manual_seed(456)
    tokens = torch.randint(0, vocab_size, (batch, seqlen), device=device)
    targets = torch.randint(0, vocab_size, (batch, seqlen), device=device)

    # Forward + loss
    logits_flash = model_flash(tokens, causal=causal)
    logits_sdpa = model_sdpa(tokens, causal=causal)

    loss_flash = F.cross_entropy(logits_flash.view(-1, vocab_size).float(), targets.view(-1))
    loss_sdpa = F.cross_entropy(logits_sdpa.view(-1, vocab_size).float(), targets.view(-1))

    loss_err = abs(loss_flash.item() - loss_sdpa.item())
    logit_err = (logits_flash.float() - logits_sdpa.float()).abs().max().item()

    # Backward
    loss_flash.backward()
    loss_sdpa.backward()

    # Compare embedding gradients
    embed_grad_err = (model_flash.embed.weight.grad.float() - model_sdpa.embed.weight.grad.float()).abs().max().item()

    # Compare all weight gradients
    max_wgrad_err = 0.0
    worst_param = ""
    for (n1, p1), (n2, p2) in zip(model_flash.named_parameters(), model_sdpa.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            err = (p1.grad.float() - p2.grad.float()).abs().max().item()
            if err > max_wgrad_err:
                max_wgrad_err = err
                worst_param = n1

    tag = "causal" if causal else "noncsl"
    ok = "OK" if max(loss_err, logit_err, embed_grad_err, max_wgrad_err) < 0.1 else "FAIL"
    print(f"  B={batch} sl={seqlen:4d} d={d_model:4d} h={nheads} L={nlayers} {tag}:")
    print(f"    loss={loss_err:.6f}  logits={logit_err:.6f}  embed_grad={embed_grad_err:.6f}  max_dW={max_wgrad_err:.6f} ({worst_param})  {ok}")


if __name__ == '__main__':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    for causal in [False, True]:
        for d_model, nheads in [(128, 4), (256, 4), (512, 8)]:
            for seqlen in [64, 128, 256]:
                compare(batch=2, seqlen=seqlen, d_model=d_model, nheads=nheads,
                        causal=causal, nlayers=2)
        print()
