import torch
from flash_attn_interface import flash_attn_func
from torch import nn


class EfficienctMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0, use_flash_attn=True):
        super().__init__()
        assert embed_size % num_heads == 0, f"{embed_size=} {num_heads=}"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.use_flash_attn = use_flash_attn and (flash_attn_func is not None)

        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size)
        self.out_proj = nn.Linear(embed_size, embed_size)
        self.dropout = dropout

    def forward(self, x, attention_mask=None):
        N, seq_length, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(N, seq_length, self.num_heads, self.head_dim)
        k = k.view(N, seq_length, self.num_heads, self.head_dim)
        v = v.view(N, seq_length, self.num_heads, self.head_dim)

        if self.use_flash_attn and attention_mask is None:
            out = flash_attn_func(
                q, k, v
            )
        out = out.reshape(N, seq_length, self.embed_size)
        out = self.out_proj(out)
        return out


def create_model(batch_size=16, sequence_length=256, embedding_dim=2048, num_heads=16):
    model = EfficienctMultiHeadAttention(embedding_dim, num_heads).cuda().bfloat16()
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim).cuda().bfloat16()
    return model, input_tensor


def test_export_model():
    model, input_tensor = create_model()
    expected = torch.compile(model, backend="aot_eager")(input_tensor)
    loss = expected.sum()
    loss.backward()

    ep = torch.export.export(model, (input_tensor,))
    got = ep.module()(input_tensor,)
    assert torch.equal(expected, got)

    loss_2 = got.sum()
    loss_2.backward()

    assert torch.equal(loss, loss_2)


def test_compile_and_package_model():
    model, input_tensor = create_model()
    expected = torch.compile(model, backend="aot_eager")(input_tensor)

    exported = torch.export.export(model, (input_tensor,))
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path="model.pt2",
    )

    compiled_model = torch._inductor.package.load_package("model.pt2")
    out = compiled_model(input_tensor,)
    assert torch.equal(expected, out)
