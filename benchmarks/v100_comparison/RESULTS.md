# V100 Attention Backend ベンチマーク結果

## 環境
- **GPU**: Tesla V100-SXM2-32GB
- **PyTorch**: 2.6.0a0 (NGC 24.11)
- **dtype**: FP16, causal=True, dropout=0.0

## V100での各バックエンド実態

| Backend | V100で使用される実装 | 備考 |
|---------|---------------------|------|
| flash_attn | Flash Attention v2 (SM70 MMA atom) | 本リポジトリのカスタムビルド |
| PyTorch SDPA | memory-efficient attention | flash backendはSM80+のみ |
| xFormers | CUTLASS backend | flash backendはSM80+のみ |

## 速度比較

| モデル | seqlen | batch | Backend | Fwd(ms) | Bwd(ms) | Total(ms) |
|--------|--------|-------|---------|---------|---------|-----------|
| GPT-small (124M) | 512 | 16 | **flash_attn** | 54.7 | **106.6** | **161.2** |
| | | | pytorch_sdpa | **54.0** | 107.5 | 161.6 |
| | | | xformers | 54.2 | 107.7 | 161.9 |
| GPT-small | 1024 | 8 | **flash_attn** | 57.7 | **115.5** | **173.2** |
| | | | pytorch_sdpa | **56.5** | 118.0 | 174.5 |
| | | | xformers | 56.7 | 118.2 | 174.8 |
| GPT-small | 2048 | 4 | **flash_attn** | 64.6 | **133.1** | **197.7** |
| | | | pytorch_sdpa | **62.0** | 137.6 | 199.5 |
| | | | xformers | 62.3 | 137.6 | 199.9 |
| GPT-medium (350M) | 512 | 8 | flash_attn | 59.8 | 131.5 | 191.3 |
| | | | pytorch_sdpa | **58.5** | 132.6 | **191.1** |
| | | | xformers | 58.6 | 132.6 | 191.3 |
| GPT-medium | 1024 | 4 | flash_attn | 64.3 | **143.9** | **208.2** |
| | | | pytorch_sdpa | **62.1** | 146.5 | 208.6 |
| | | | xformers | 62.3 | 146.5 | 208.8 |
| GPT-medium | 2048 | 2 | flash_attn | 73.7 | **168.1** | **241.8** |
| | | | pytorch_sdpa | **69.4** | 173.2 | 242.6 |
| | | | xformers | 69.6 | 173.8 | 243.4 |

## メモリ比較

| Backend | GPT-small PeakMem(MB) | GPT-medium PeakMem(MB) |
|---------|----------------------|------------------------|
| flash_attn | 8,494 | 7,424 |
| pytorch_sdpa | 8,494 | 7,424 |
| xformers | 8,811 (+317) | 8,206 (+782) |

## 精度比較 (flash_attn基準)

| Backend | Logit最大誤差 | 勾配最大誤差 |
|---------|-------------|------------|
| pytorch_sdpa | 0.003~0.005 | 0.00002~0.00004 |
| xformers | 0.003~0.005 | 0.00002~0.00004 |

## 考察

1. **速度**: 3つのバックエンドはV100上でほぼ同等（差は~1%以内）。flash_attnはbackwardが3-5ms速く、SDPAはforwardが2-4ms速い。トータルではflash_attnがわずかに優位。

2. **メモリ**: flash_attnとSDPAは同一のピークメモリ。xFormersは300~800MB多く使用。

3. **精度**: SDPAとxFormersは同一の誤差を示す（V100ではどちらもmemory-efficient attention実装を使用するため）。誤差は極めて小さく実用上問題なし。

## 再現方法

```bash
# 1. ベースイメージビルド（初回のみ、約2時間）
docker build -t flash_attn_v100_base -f benchmarks/v100_comparison/Dockerfile.base .

# 2. ベンチマークイメージビルド（xFormers + スクリプト）
docker build -t flash_attn_bench -f benchmarks/v100_comparison/Dockerfile .

# 3. ベンチマーク実行
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py

# dry-run（動作確認のみ）
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py --dry-run

# 特定バックエンドのみ
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py --backends flash_attn pytorch_sdpa
```
