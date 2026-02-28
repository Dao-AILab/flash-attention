# FlashAttention V100 (SM70) ベンチマークレポート

## 1. 概要

FlashAttention v2にV100 (SM70/Volta) サポートを追加し、既存のattentionバックエンド（PyTorch SDPA、xFormers）との性能を比較した。ベンチマークは2種類実施した。

| ベンチマーク | 目的 | 比較対象 |
|-------------|------|---------|
| **Training (GPTモデル)** | forward/backward カーネル単体の速度・メモリ | flash_attn, PyTorch SDPA, xFormers |
| **Serving (vLLM推論)** | 実運用サーバの TTFT・スループット | flash_attn, xFormers |

## 2. V100上のAttention実装の整理

V100 (SM70) では、表面上3つのバックエンド名が存在するが、実際に動作するカーネル実装は2種類に集約される。

```
呼び出しAPI                    V100での実カーネル
─────────────────────────────────────────────────────
flash_attn (本リポジトリ)  →  Flash Attention v2 (SM70 MMA atom)
PyTorch SDPA               →  memory-efficient attention ─┐ 同一カーネル
xFormers                   →  memory-efficient attention ─┘ (CUTLASS)
```

| API | V100でdispatchされる実装 | 理由 |
|-----|------------------------|------|
| flash_attn | **Flash Attention v2 (SM70)** | 本リポジトリのカスタムビルド |
| PyTorch SDPA (`F.scaled_dot_product_attention`) | **memory-efficient attention** | SDPA内部の flash backend は SM80+ 専用。V100では自動的に mem-efficient にfallback |
| xFormers (`xformers.ops.memory_efficient_attention`) | **memory-efficient attention** | xFormersの flash backend も SM80+ 専用。V100ではCUTLASS backend を使用 |

PyTorch SDPAとxFormersはV100上で同一のCUTLASSカーネルを呼ぶため、Trainingベンチマークで速度・精度がほぼ一致する（セクション4参照）。

Servingベンチマーク（セクション5）では、vLLM v0.6.5の`TORCH_SDPA`バックエンドがCPU専用のため除外した。ただし上記の通り、xFormers（CUTLASS）の結果がPyTorch SDPAと実質同等であるため、2バックエンド（FLASH_ATTN, XFORMERS）で3つのAPIすべてをカバーしている。

## 3. 環境

| 項目 | 値 |
|------|-----|
| GPU | Tesla V100-SXM2-32GB |
| PyTorch | 2.6.0a0+df5bbc0 (NGC 24.11) |
| CUDA | 12.6 |
| dtype | FP16 |
| flash_attn | v2 + SM70カスタムビルド |
| xFormers | v0.0.29.post2 (CUTLASS backend) |
| vLLM | v0.6.5 (ソースビルド) |

## 4. Training ベンチマーク（GPTモデル forward/backward）

GPT-small (124M) および GPT-medium (350M) で、異なるシーケンス長（512/1024/2048）におけるforward/backward速度、ピークメモリ使用量、数値精度を計測。

### 4.1 各バックエンドのV100上での実装

| Backend | V100での実装 | 備考 |
|---------|-------------|------|
| flash_attn | Flash Attention v2 (SM70 MMA atom) | 本リポジトリのカスタムビルド |
| PyTorch SDPA | memory-efficient attention | flash backendはSM80+のみ |
| xFormers | CUTLASS backend | flash backendはSM80+のみ |

### 4.2 速度比較

#### GPT-small (124M, nhead=12, hdim=64)

| seqlen | batch | Backend | Fwd (ms) | Bwd (ms) | Total (ms) | vs flash_attn |
|--------|-------|---------|----------|----------|------------|---------------|
| 512 | 16 | **flash_attn** | 54.4 | **106.5** | **160.9** | - |
| | | pytorch_sdpa | **53.8** | 107.5 | 161.3 | +0.2% |
| | | xformers | 54.0 | 107.8 | 161.9 | +0.6% |
| 1024 | 8 | **flash_attn** | 57.7 | **115.5** | **173.2** | - |
| | | pytorch_sdpa | **56.5** | 118.0 | 174.4 | +0.7% |
| | | xformers | 56.7 | 118.4 | 175.1 | +1.1% |
| 2048 | 4 | **flash_attn** | 64.3 | **133.1** | **197.4** | - |
| | | pytorch_sdpa | **61.9** | 137.5 | 199.3 | +1.0% |
| | | xformers | 61.7 | 137.6 | 199.3 | +1.0% |

#### GPT-medium (350M, nhead=16, hdim=64)

| seqlen | batch | Backend | Fwd (ms) | Bwd (ms) | Total (ms) | vs flash_attn |
|--------|-------|---------|----------|----------|------------|---------------|
| 512 | 8 | flash_attn | 59.4 | **131.4** | 190.7 | - |
| | | pytorch_sdpa | **58.1** | 132.9 | 191.0 | +0.2% |
| | | xformers | 58.1 | 132.6 | **190.7** | 0.0% |
| 1024 | 4 | flash_attn | 63.9 | **143.8** | **207.7** | - |
| | | pytorch_sdpa | **61.8** | 146.3 | 208.1 | +0.2% |
| | | xformers | 62.3 | 146.3 | 208.6 | +0.4% |
| 2048 | 2 | flash_attn | 73.5 | **167.9** | **241.5** | - |
| | | pytorch_sdpa | **69.1** | 173.3 | 242.4 | +0.4% |
| | | xformers | 69.2 | 173.2 | 242.4 | +0.4% |

### 4.3 メモリ使用量

| Backend | GPT-small PeakMem (MB) | GPT-medium PeakMem (MB) |
|---------|----------------------|------------------------|
| flash_attn | 8,494 | 7,424 |
| pytorch_sdpa | 8,494 | 7,424 |
| xformers | 8,811 (+3.7%) | 8,206 (+10.5%) |

### 4.4 数値精度（flash_attn基準）

| Backend | Logit 最大誤差 | 勾配 最大誤差 |
|---------|--------------|-------------|
| pytorch_sdpa | 0.003~0.005 | 0.00002~0.00004 |
| xformers | 0.003~0.005 | 0.00002~0.00004 |

### 4.5 Training ベンチマーク まとめ

- **速度**: 3バックエンドはV100上でほぼ同等（差は1%以内）。flash_attnはbackwardで3-5ms速く、SDPAはforwardで2-4ms速い。**トータルではflash_attnがわずかに最速**。
- **メモリ**: flash_attnとSDPAは同一のピークメモリ。xFormersは300~800MB多く使用。
- **精度**: すべてのバックエンドで実用上問題のない精度。

---

## 5. Serving ベンチマーク（vLLM推論サーバ）

vLLM v0.6.5でLlama-2-7B (FP16) を提供し、OpenAI互換APIへ非同期ストリーミングリクエストを送信して計測。

> **PyTorch SDPA (TORCH_SDPA) について**: vLLM v0.6.5の`TORCH_SDPA`バックエンドはCPU専用のため本ベンチマークから除外した。ただしセクション2で説明の通り、V100上のPyTorch SDPAとxFormersは同一のCUTLASSカーネルにdispatchされるため、XFORMERSの結果がPyTorch SDPAの性能も代表する。

### 5.1 計測条件

| 項目 | 値 |
|------|-----|
| モデル | NousResearch/Llama-2-7b-hf (FP16) |
| max_model_len | 4096 |
| gpu_memory_utilization | 0.9 |
| 推論モード | eager (torch.compile無効) |
| FLASH_ATTN block_size | 256（SM70 paged KV cache制約） |
| XFORMERS block_size | 16（デフォルト） |
| リクエスト数/シナリオ | 32 |

### 5.2 ワークロード定義

| ワークロード | 入力長 | 最大出力長 | 想定ユースケース |
|-------------|--------|-----------|----------------|
| short | 128 | 128 | チャット応答 |
| medium | 512 | 256 | 要約 |
| long | 1024 | 512 | 長文生成 |
| very_long | 2048 | 256 | 長文理解 |

同時接続数: 1, 2, 4, 8, 16

### 5.3 総合結果

| メトリクス | FLASH_ATTN (SM70) | XFORMERS | XFORMERS 優位率 |
|-----------|-------------------|----------|----------------|
| 平均 TTFT (初回トークン遅延) | 1,200 ms | 862 ms | 1.39x 高速 |
| 平均 TPOT (トークン生成間隔) | 57.1 ms | 32.9 ms | 1.74x 高速 |
| 平均スループット | 91.3 tok/s | 155.7 tok/s | 1.71x 高スループット |

### 5.4 詳細結果 — FLASH_ATTN (SM70)

| ワークロード | 同時接続 | 成功率 | TTFT avg (ms) | TTFT p99 (ms) | TPOT (ms) | スループット (tok/s) | Req/s |
|-------------|---------|--------|--------------|--------------|----------|-------------------|-------|
| short | 1 | 32/32 | 62 | 73 | 27.1 | 36.5 | 0.29 |
| short | 2 | 32/32 | 90 | 123 | 29.1 | 67.5 | 0.53 |
| short | 4 | 32/32 | 232 | 1,223 | 33.7 | 113.5 | 0.89 |
| short | 8 | 32/32 | 209 | 250 | 41.4 | 187.1 | 1.46 |
| short | 16 | 32/32 | 1,042 | 1,847 | 56.5 | 249.3 | 1.95 |
| medium | 1 | 32/32 | 114 | 116 | 30.0 | 33.0 | 0.13 |
| medium | 2 | 32/32 | 171 | 229 | 32.7 | 60.1 | 0.23 |
| medium | 4 | 32/32 | 336 | 410 | 40.7 | 95.5 | 0.37 |
| medium | 8 | 32/32 | 697 | 784 | 56.4 | 135.8 | 0.53 |
| medium | 16 | 32/32 | 1,299 | 1,994 | 82.7 | 182.9 | 0.71 |
| long | 1 | 32/32 | 220 | 224 | 35.3 | 28.0 | 0.05 |
| long | 2 | 32/32 | 329 | 441 | 38.0 | 51.8 | 0.10 |
| long | 4 | 32/32 | 678 | 836 | 50.1 | 77.9 | 0.15 |
| long | 8 | 32/32 | 1,098 | 1,670 | 75.2 | 103.7 | 0.20 |
| long | 16 | 32/32 | 2,170 | 3,721 | 114.7 | 134.7 | 0.26 |
| very_long | 1 | 32/32 | 459 | 462 | 42.9 | 22.4 | 0.09 |
| very_long | 2 | 32/32 | 689 | 924 | 46.3 | 41.0 | 0.16 |
| very_long | 4 | 32/32 | 1,147 | 1,840 | 66.5 | 56.5 | 0.22 |
| very_long | 8 | 32/32 | 2,062 | 3,674 | 107.6 | 69.4 | 0.27 |
| very_long | 16 | 32/32 | 10,885 | 41,150 | 134.1 | 78.9 | 0.31 |

### 5.5 詳細結果 — XFORMERS (CUTLASS)

| ワークロード | 同時接続 | 成功率 | TTFT avg (ms) | TTFT p99 (ms) | TPOT (ms) | スループット (tok/s) | Req/s |
|-------------|---------|--------|--------------|--------------|----------|-------------------|-------|
| short | 1 | 32/32 | 60 | 73 | 25.3 | 39.2 | 0.31 |
| short | 2 | 32/32 | 90 | 120 | 28.0 | 70.2 | 0.55 |
| short | 4 | 32/32 | 131 | 163 | 28.6 | 136.0 | 1.06 |
| short | 8 | 32/32 | 218 | 249 | 29.5 | 257.9 | 2.01 |
| short | 16 | 32/32 | 401 | 455 | 31.8 | 460.4 | 3.60 |
| medium | 1 | 32/32 | 110 | 112 | 28.2 | 35.0 | 0.14 |
| medium | 2 | 32/32 | 165 | 222 | 29.0 | 67.7 | 0.26 |
| medium | 4 | 32/32 | 329 | 402 | 29.3 | 131.1 | 0.51 |
| medium | 8 | 32/32 | 1,182 | 3,039 | 31.6 | 221.8 | 0.87 |
| medium | 16 | 32/32 | 1,072 | 1,540 | 37.8 | 382.5 | 1.49 |
| long | 1 | 32/32 | 209 | 211 | 27.6 | 35.8 | 0.07 |
| long | 2 | 32/32 | 314 | 421 | 28.5 | 68.8 | 0.13 |
| long | 4 | 32/32 | 647 | 798 | 29.5 | 130.1 | 0.25 |
| long | 8 | 32/32 | 1,131 | 1,989 | 34.7 | 217.0 | 0.42 |
| long | 16 | 32/32 | 1,859 | 3,130 | 45.0 | 329.3 | 0.64 |
| very_long | 1 | 32/32 | 419 | 421 | 27.6 | 34.3 | 0.13 |
| very_long | 2 | 32/32 | 629 | 839 | 29.3 | 63.2 | 0.25 |
| very_long | 4 | 32/32 | 1,048 | 1,671 | 33.9 | 105.6 | 0.41 |
| very_long | 8 | 32/32 | 1,882 | 3,342 | 44.6 | 154.6 | 0.60 |
| very_long | 16 | 32/32 | 5,342 | 19,279 | 57.6 | 173.8 | 0.68 |

### 5.6 ワークロード別スループット比較

| ワークロード | 同時接続 | FLASH_ATTN (tok/s) | XFORMERS (tok/s) | XFORMERS / FLASH_ATTN |
|-------------|---------|-------------------|-----------------|----------------------|
| short | 1 | 36.5 | 39.2 | 1.07x |
| short | 16 | 249.3 | 460.4 | **1.85x** |
| medium | 1 | 33.0 | 35.0 | 1.06x |
| medium | 16 | 182.9 | 382.5 | **2.09x** |
| long | 1 | 28.0 | 35.8 | 1.28x |
| long | 16 | 134.7 | 329.3 | **2.44x** |
| very_long | 1 | 22.4 | 34.3 | 1.53x |
| very_long | 16 | 78.9 | 173.8 | **2.20x** |

### 5.7 Serving ベンチマーク分析

**同時接続数が増えるほどXFORMERSの優位が拡大する**（1接続で1.1~1.5x → 16接続で1.9~2.4x）。原因はKV cacheのblock sizeの差にある。

| 要因 | FLASH_ATTN | XFORMERS |
|------|-----------|----------|
| KV cache block_size | 256 | 16 |
| ブロックあたり割当量 | 256トークン分 | 16トークン分 |
| 部分使用時の無駄 | 大きい | 小さい |

- **block_size=256の制約**: SM70のflash_attnはページテーブルの次元がattentionブロックサイズの倍数である必要があり、block_size=256が最小値。1ブロックに256トークン分のメモリが確保されるため、短いシーケンスでもメモリが無駄になる。
- **continuous batching への影響**: メモリ効率が悪いため、同時処理可能なシーケンス数が制限される。高同時接続数で差が拡大するのはこのため。
- **カーネル速度自体は同等**: Trainingベンチマークが示す通り、attention カーネル単体の速度差はほぼない。Servingでの差はシステムレベルのメモリ管理に起因する。

---

## 6. 総合考察

### 6.1 ユースケース別推奨

| ユースケース | 推奨バックエンド | 理由 |
|-------------|----------------|------|
| **Training (学習)** | **flash_attn** | カーネル速度が最速（backward 3-5ms高速）、メモリ効率最良 |
| **Serving (推論、低同時接続)** | flash_attn or xFormers | 差は小さい（1.1~1.5x） |
| **Serving (推論、高同時接続)** | **xFormers** | block_size=16のメモリ効率が有利（1.9~2.4x高速） |

### 6.2 flash_attn SM70の価値

1. **Training**: flash_attnはV100でのtraining用途において最良の選択肢。paged KV cacheを使わないためblock_size制約が無関係で、backward速度が最速。

2. **Serving**: vLLM上での推論は動作するが、`block_size=256`制約により高同時接続時のスループットがxFormersに劣る。低同時接続（1-2）では差は小さい。

3. **互換性**: flash_attn SM70サポートにより、V100ユーザーがflash_attnに依存するコードベース（HuggingFace Transformers等）をV100上で動作させることが可能になる。

### 6.3 制限事項

| 制限 | 詳細 |
|------|------|
| FP16のみ | BF16はVoltaハードウェア非対応 |
| Dropout非対応 | SM70での検証未実施のためブロック |
| hdim=256 causal backward | V100の96KB共有メモリ上限を超過 |
| SplitKV | SM70では単一splitに制限 |
| Serving block_size | 256固定（xFormersの16と比較して非効率） |

---

## 7. 再現方法

### 7.1 Training ベンチマーク

```bash
# ベースイメージビルド（初回のみ、約2時間）
docker build -t flash_attn_v100_base -f benchmarks/v100_comparison/Dockerfile.base .

# ベンチマークイメージビルド（xFormers + スクリプト）
docker build -t flash_attn_bench -f benchmarks/v100_comparison/Dockerfile .

# 実行
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py

# dry-run
docker run --gpus '"device=0"' --rm flash_attn_bench python benchmark_gpt.py --dry-run
```

### 7.2 Serving ベンチマーク

```bash
# vLLMイメージビルド（flash_attn_benchベース）
docker build -t flash_attn_v100_vllm -f benchmarks/v100_vllm_serving/Dockerfile .

# 実行（HFキャッシュをマウントしてモデル再ダウンロード回避）
docker run --gpus '"device=0"' --ipc=host --rm \
    -e PYTHONUNBUFFERED=1 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    flash_attn_v100_vllm python benchmark_serving.py

# dry-run
docker run --gpus '"device=0"' --ipc=host --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    flash_attn_v100_vllm python benchmark_serving.py --dry-run
```

---

## 8. ファイル構成

```
benchmarks/
├── v100_comparison/                    # Training ベンチマーク
│   ├── Dockerfile.base                 # NGC PyTorch + flash_attn SM70ビルド
│   ├── Dockerfile                      # + xFormers + ベンチマークスクリプト
│   ├── benchmark_gpt.py                # GPTモデル fwd/bwd 計測スクリプト
│   └── RESULTS.md                      # Training ベンチマーク結果
├── v100_vllm_serving/                  # Serving ベンチマーク
│   ├── Dockerfile                      # vLLM v0.6.5 + パッチ
│   ├── benchmark_serving.py            # vLLM推論サーバ計測スクリプト
│   ├── vllm_flash_attn_shim.py         # vLLM ↔ flash_attn API ブリッジ
│   └── RESULTS.md                      # Serving ベンチマーク結果
└── V100_BENCHMARK_REPORT.md            # 本レポート（総合）
```
