# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import argparse
import math
import torch
import csv

import os
from datetime import date
import subprocess

from flash_attn.utils.benchmark import benchmark_forward, benchmark_backward
from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func


def benchmark_row(row):
    dtype = row["dtype"]
    if dtype in ["torch.float16"]:
        dtype = torch.float16
    elif dtype in ["torch.bfloat16"]:
        dtype = torch.bfloat16
    else:
        raise ValueError("Wrong data type")
    batch_size = row["batch size"]
    nheads = int(row["nheads"])
    d = int(row["embedding dim"])
    seqlen = int(row["seqlen"])
    causal = row["causal"] == 'TRUE'
    dropout_p = float(row["dropout"])

    torch.manual_seed(0)
    
    if not batch_size.isdigit():
        print(dtype, batch_size, seqlen, nheads, d, causal, dropout_p)
        cu_seqlens = [int(b) for b in batch_size.split(',')]
        max_seqlen = 0
        for cu_seq1, cu_seq2 in zip(cu_seqlens[1:], cu_seqlens[:-1]):
            max_seqlen = max(max_seqlen, cu_seq1 - cu_seq2)
        qkv = torch.randn(seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
        fn = lambda qkv: flash_attn_varlen_qkvpacked_func(
            qkv, torch.tensor(cu_seqlens, dtype=torch.int32).cuda(), max_seqlen, dropout_p, causal=causal, softmax_scale=1/math.sqrt(d)
        )
    else:
        print(dtype, batch_size, seqlen, nheads, d, causal, dropout_p)
        batch_size = int(batch_size)
        qkv = torch.randn(batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True)
        fn = lambda qkv: flash_attn_qkvpacked_func(
            qkv, dropout_p, causal=causal, softmax_scale=1/math.sqrt(d)
        )

    _, m1 = benchmark_forward(fn, qkv, amp_dtype=dtype, repeats=repeats, verbose=False, desc='FlashAttention')
    _, m2 = benchmark_backward(fn, qkv, amp_dtype=dtype, repeats=repeats, verbose=False, desc='FlashAttention')
    
    fwd_time = m1.mean
    bwd_time = m2.mean
    if isinstance(batch_size, str):
        batch_size = 1
    fwd_tflops = efficiency(flops(batch_size, seqlen, d, nheads, causal, mode="fwd"), fwd_time)
    bwd_tflops = efficiency(flops(batch_size, seqlen, d, nheads, causal, mode="bwd"), bwd_time)
    fwd_bwd_tflops = efficiency(flops(batch_size, seqlen, d, nheads, causal, mode="fwd_bwd"), fwd_time+bwd_time)
    return [dtype, batch_size, nheads, d, seqlen, causal, dropout_p, format(fwd_time*1000, ".2f"), format(bwd_time*1000, ".2f"), format(fwd_tflops, ".2f"), format(bwd_tflops, ".2f"), format(fwd_bwd_tflops, ".2f")]


def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark flash attention.")

    parser.add_argument("--repeats",
                        type=int,
                        default=30)
    parser.add_argument("--output_format",
                        type=str,
                        default='csv',
                        choices=['csv', 'xls'],
                        help="Export file format")
    parser.add_argument("--input_csv",
                    type=str,
                    required=True,
                    help="Input csv path")
    parser.add_argument("--output_csv",
                    type=str,
                    required=True,
                    help="Output csv path")
    
    args = parser.parse_args()

    fa_commit = subprocess.run("git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
    submodule = subprocess.run("git submodule foreach", shell=True, capture_output=True).stdout.strip().decode('UTF-8')
    ck_path = submodule.split(' ')[1][1:-1]
    ck_commit = subprocess.run(f"cd {ck_path} && git rev-parse HEAD", shell=True, capture_output=True).stdout.strip().decode('UTF-8')

    datetime = date.today()
    labels = ["dtype", "batch size", "seqlen", "nheads", "embedding dim", "causal", "dropout", "fwd(ms)", "bwd(ms)", "fwd(tflops)", "bwd(tflops)", "fwd+bwd(tflops)"]
    device = 'cuda'
    
    repeats = args.repeats
    with open(args.input_csv, newline='') as input_csv:
        csvreader = csv.DictReader(input_csv)
        if args.output_format == 'xls':
            import xlwt
            workbook = xlwt.Workbook(encoding = 'utf-8')
            worksheet = workbook.add_sheet('flash attention')
            
            for i, label in enumerate(labels):
                worksheet.write(0, i, label)
            
            i = 1
            for row in csvreader:
                output_row = benchmark_row(row)
                for j, value in enumerate(output_row):
                    worksheet.write(i, j, str(value))
                i += 1

            workbook.save(args.output_csv)
        else:
            with open(args.output_csv, 'w', newline='') as output_csv:
                output_csv = csv.writer(output_csv, delimiter=',')
                output_csv.writerow(labels)
                output_csv.writerows([benchmark_row(row) for row in csvreader])