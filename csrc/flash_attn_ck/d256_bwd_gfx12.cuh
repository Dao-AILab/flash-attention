// Head-dimension-split ("D-split") backward for D = 256, bf16, gfx12 (RDNA4).
//
// Three kernels, launched by fa_d256::run():
//   dot_do_o : D[b,h,i] = sum_d dO * O                                     (softmax_d output)
//   dq       : WG = 4 waves, 32 query rows; loops over 16-key tiles
//   dkdv     : WG = 4 waves, 16 keys;        loops over 32-row query tiles
// In dq/dkdv wave w owns head-dim columns [64w, 64w+64) of every operand and accumulator.
// S = Q K^T and dP = dO V^T are computed as per-wave partial sums over the wave's 64 columns,
// reduced through LDS (f32); P / dS are formed once per element and exchanged as bf16 tiles.
// Wave32 v_wmma_f32_16x16x16_bf16 operand layout:
//   A: lane l -> row l%16, k 8*(l/16)+j     B: lane l -> col l%16, k 8*(l/16)+j
//   C: lane l -> col l%16, row 8*(l/16)+j
//   global_load_tr_b128: each 8-lane group transposes one 8x8 block; lane i of group g supplies
//   row row0 + i + 8*(g/2), cols col0 + 8*(g%2).. -> B fragment of a row-major [k][n] tile.
// Masks: none, or bottom-right causal (key j visible to query i iff j <= i + Sk - Sq).
// Rows with lse == -inf (no visible key) produce zero gradients.
#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

// Device pass for an RDNA4 target (clang defines __gfx1200__/__gfx1201__;
// __gfx12__ is a CK-only macro).
#if defined(__gfx1200__) || defined(__gfx1201__)
#define FA_D256_GFX12_DEVICE 1
#endif

namespace fa_d256 {

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 llvm_fp16x8_t;

constexpr int D = 256, DW = 64, NWAVE = 4;
constexpr int QM0 = 32, QN0 = 16; // dq kernel tile
constexpr int KM0 = 32, KN0 = 16; // dkdv kernel tile

struct Args {
    const __bf16 *q, *k, *v, *dout, *o;
    const float *lse;
    float *dvec;
    __bf16 *dq, *dk, *dv; // dk / dv are per query head (GQA expanded)
    int64_t q_b, q_s, q_h, k_b, k_s, k_h, v_b, v_s, v_h, do_b, do_s, do_h, o_b, o_s, o_h;
    int64_t dq_b, dq_s, dq_h, dk_b, dk_s, dk_h, dv_b, dv_s, dv_h;
    int64_t lse_b, lse_h, dvec_b, dvec_h;
    int B, H, Hk, Sq, Sk;
    float scale;
    int causal;
};

#if defined(FA_D256_GFX12_DEVICE)
__device__ inline f32x8 wmma(const bf16x8 &a, const bf16x8 &b, const f32x8 &c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
__device__ inline bf16x8 zero8() {
    bf16x8 z;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        z[j] = (__bf16)0.f;
    return z;
}
__device__ inline bf16x8 ld8(const __bf16 *base, int64_t rs, int row, int nrow, int col) {
    if (row >= nrow)
        return zero8();
    return *reinterpret_cast<const bf16x8 *>(base + row * rs + col);
}
// B fragment of a row-major [k][n] tile at (row0, col0) via
// global_load_tr_b128. Rows >= nrow are clamped to nrow-1 (finite data that
// callers multiply by zero).
__device__ inline bf16x8 ldtr(const __bf16 *base, int64_t rs, int row0, int nrow, int col0,
                              int lane) {
    const int g = lane / 8, i = lane % 8;
    const int row = min(row0 + i + 8 * (g / 2), nrow - 1);
    const __bf16 *p = base + row * rs + col0 + 8 * (g % 2);
    auto *gp = reinterpret_cast<__attribute__((address_space(1))) llvm_fp16x8_t *>(
        reinterpret_cast<uintptr_t>(p));
    const llvm_fp16x8_t v = __builtin_amdgcn_global_load_tr_b128_v8f16(gp);
    return __builtin_bit_cast(bf16x8, v);
}
// LDS-only workgroup barrier (__syncthreads would also invalidate the vector L0
// via global_inv).
__device__ inline void lds_barrier() {
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    asm volatile("s_barrier_signal -1" ::: "memory");
    asm volatile("s_barrier_wait -1" ::: "memory");
}
__device__ inline float lse_log2(float l) { return l == -INFINITY ? 0.f : l * 1.4426950408889634f; }
#endif

// ---------------------------------------------------------------- D =
// rowsum(dO * O)
__global__ __launch_bounds__(128) void dot_do_o_kernel(Args a) {
#if defined(FA_D256_GFX12_DEVICE)
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + w, h = blockIdx.y, b = blockIdx.z;
    if (i >= a.Sq)
        return;
    const bf16x8 x =
        *reinterpret_cast<const bf16x8 *>(a.dout + b * a.do_b + i * a.do_s + h * a.do_h + 8 * lane);
    const bf16x8 y =
        *reinterpret_cast<const bf16x8 *>(a.o + b * a.o_b + i * a.o_s + h * a.o_h + 8 * lane);
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        s += (float)x[j] * (float)y[j];
#pragma unroll
    for (int off = 16; off > 0; off /= 2)
        s += __shfl_xor(s, off, 32);
    if (lane == 0)
        a.dvec[b * a.dvec_b + h * a.dvec_h + i] = s;
#endif
}

// ---------------------------------------------------------------- dQ
__global__ __launch_bounds__(128) void dq_kernel(Args a) {
#if defined(FA_D256_GFX12_DEVICE)
    __shared__ float s_part[NWAVE][2][QM0][QN0];
    __shared__ __bf16 s_ds[QM0][QN0 + 8];

    const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int q0 = blockIdx.x * QM0, h = blockIdx.y, b = blockIdx.z, hk = h / (a.H / a.Hk);
    const int Sq = a.Sq, Sk = a.Sk;
    const __bf16 *qb = a.q + b * a.q_b + h * a.q_h;
    const __bf16 *ob = a.dout + b * a.do_b + h * a.do_h;
    const __bf16 *kb = a.k + b * a.k_b + hk * a.k_h;
    const __bf16 *vb = a.v + b * a.v_b + hk * a.v_h;
    const float *lseb = a.lse + b * a.lse_b + h * a.lse_h;
    const float *db = a.dvec + b * a.dvec_b + h * a.dvec_h;
    const int dcol0 = w * DW, lr = lane % 16, lk = 8 * (lane / 16);
    const float scale_log2 = a.scale * 1.4426950408889634f;

    bf16x8 qa[2][4], oa[2][4];
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            qa[mb][c] = ld8(qb, a.q_s, q0 + 16 * mb + lr, Sq, dcol0 + 16 * c + lk);
            oa[mb][c] = ld8(ob, a.do_s, q0 + 16 * mb + lr, Sq, dcol0 + 16 * c + lk);
        }
    f32x8 acc[2][4];
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
        for (int c = 0; c < 4; ++c)
            acc[mb][c] = f32x8{};

    int k_end = Sk;
    if (a.causal) {
        const int last_q = min(q0 + QM0, Sq) - 1;
        k_end = min(Sk, max(0, last_q + Sk - Sq + 1));
    }

    for (int n0 = 0; n0 < k_end; n0 += QN0) {
        bf16x8 kbf[4], vbf[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            kbf[c] = ld8(kb, a.k_s, n0 + lr, Sk, dcol0 + 16 * c + lk);
            vbf[c] = ld8(vb, a.v_s, n0 + lr, Sk, dcol0 + 16 * c + lk);
        }
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            f32x8 sp = f32x8{}, dpp = f32x8{};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                sp = wmma(qa[mb][c], kbf[c], sp);
                dpp = wmma(oa[mb][c], vbf[c], dpp);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                s_part[w][0][16 * mb + lk + j][lr] = sp[j];
                s_part[w][1][16 * mb + lk + j][lr] = dpp[j];
            }
        }
        lds_barrier();
        {
            const int r = 8 * w + lane / 4, c0 = 4 * (lane % 4), qi = q0 + r;
            float l2 = 0.f, dval = 0.f;
            if (qi < Sq) {
                l2 = lse_log2(lseb[qi]);
                dval = db[qi];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kj = n0 + c0 + j;
                float s = 0.f, dp = 0.f;
#pragma unroll
                for (int ww = 0; ww < NWAVE; ++ww) {
                    s += s_part[ww][0][r][c0 + j];
                    dp += s_part[ww][1][r][c0 + j];
                }
                bool vis = (qi < Sq) && (kj < Sk);
                if (a.causal)
                    vis = vis && (kj <= qi + Sk - Sq);
                const float p = vis ? exp2f(s * scale_log2 - l2) : 0.f;
                s_ds[r][c0 + j] = (__bf16)(p * (dp - dval));
            }
        }
        lds_barrier();
        bf16x8 dsa[2];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb)
            dsa[mb] = *reinterpret_cast<const bf16x8 *>(&s_ds[16 * mb + lr][lk]);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const bf16x8 kt = ldtr(kb, a.k_s, n0, Sk, dcol0 + 16 * c, lane);
#pragma unroll
            for (int mb = 0; mb < 2; ++mb)
                acc[mb][c] = wmma(dsa[mb], kt, acc[mb][c]);
        }
        lds_barrier();
    }
    __bf16 *dqb = a.dq + b * a.dq_b + h * a.dq_h;
#pragma unroll
    for (int mb = 0; mb < 2; ++mb)
#pragma unroll
        for (int c = 0; c < 4; ++c)
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int qi = q0 + 16 * mb + lk + j;
                if (qi < Sq)
                    dqb[qi * a.dq_s + dcol0 + 16 * c + lr] = (__bf16)(acc[mb][c][j] * a.scale);
            }
#endif
}

// ---------------------------------------------------------------- dK / dV
__global__ __launch_bounds__(128) void dkdv_kernel(Args a) {
#if defined(FA_D256_GFX12_DEVICE)
    // s_part (until the reduce) and the transposed Q / dO copies (after it) share
    // one buffer.
    __shared__ __attribute__((aligned(16))) unsigned char s_big[2 * NWAVE * DW * KM0 * 2]; // 32 KB
    __shared__ __bf16 s_pt[KN0][KM0 + 16];
    __shared__ __bf16 s_dst[KN0][KM0 + 16];
    auto s_part = reinterpret_cast<float(*)[2][KM0][KN0]>(s_big);
    auto s_qt = reinterpret_cast<__bf16(*)[DW][KM0]>(s_big);
    auto s_dot = reinterpret_cast<__bf16(*)[DW][KM0]>(s_big + NWAVE * DW * KM0 * 2);
    // 16-byte chunk index XOR (d >> 2) & 3: the 16-lane B reads hit 64 distinct
    // banks.
    auto swz = [](int d, int m) { return (((m >> 3) ^ ((d >> 2) & 3)) << 3) | (m & 7); };

    const int tid = threadIdx.x, w = tid / 32, lane = tid % 32;
    const int n0 = blockIdx.x * KN0, h = blockIdx.y, b = blockIdx.z, hk = h / (a.H / a.Hk);
    const int Sq = a.Sq, Sk = a.Sk;
    const __bf16 *qb = a.q + b * a.q_b + h * a.q_h;
    const __bf16 *ob = a.dout + b * a.do_b + h * a.do_h;
    const __bf16 *kb = a.k + b * a.k_b + hk * a.k_h;
    const __bf16 *vb = a.v + b * a.v_b + hk * a.v_h;
    const float *lseb = a.lse + b * a.lse_b + h * a.lse_h;
    const float *db = a.dvec + b * a.dvec_b + h * a.dvec_h;
    const int dcol0 = w * DW, lr = lane % 16, lk = 8 * (lane / 16);
    const float scale_log2 = a.scale * 1.4426950408889634f;

    bf16x8 kbf[4], vbf[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        kbf[c] = ld8(kb, a.k_s, n0 + lr, Sk, dcol0 + 16 * c + lk);
        vbf[c] = ld8(vb, a.v_s, n0 + lr, Sk, dcol0 + 16 * c + lk);
    }
    f32x8 dk[4], dv[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        dk[c] = f32x8{};
        dv[c] = f32x8{};
    }

    int m_begin = 0;
    if (a.causal)
        m_begin = max(0, n0 - (Sk - Sq)) / KM0 * KM0;

    for (int m0 = m_begin; m0 < Sq; m0 += KM0) {
        bf16x8 qa[2][4], oa[2][4];
#pragma unroll
        for (int mb = 0; mb < 2; ++mb)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                qa[mb][c] = ld8(qb, a.q_s, m0 + 16 * mb + lr, Sq, dcol0 + 16 * c + lk);
                oa[mb][c] = ld8(ob, a.do_s, m0 + 16 * mb + lr, Sq, dcol0 + 16 * c + lk);
            }
#pragma unroll
        for (int mb = 0; mb < 2; ++mb) {
            f32x8 sp = f32x8{}, dpp = f32x8{};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                sp = wmma(qa[mb][c], kbf[c], sp);
                dpp = wmma(oa[mb][c], vbf[c], dpp);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                s_part[w][0][16 * mb + lk + j][lr] = sp[j];
                s_part[w][1][16 * mb + lk + j][lr] = dpp[j];
            }
        }
        lds_barrier();
        {
            const int r = 8 * w + lane / 4, c0 = 4 * (lane % 4), qi = m0 + r;
            float l2 = 0.f, dval = 0.f;
            if (qi < Sq) {
                l2 = lse_log2(lseb[qi]);
                dval = db[qi];
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int kj = n0 + c0 + j;
                float s = 0.f, dp = 0.f;
#pragma unroll
                for (int ww = 0; ww < NWAVE; ++ww) {
                    s += s_part[ww][0][r][c0 + j];
                    dp += s_part[ww][1][r][c0 + j];
                }
                bool vis = (qi < Sq) && (kj < Sk);
                if (a.causal)
                    vis = vis && (kj <= qi + Sk - Sq);
                const float p = vis ? exp2f(s * scale_log2 - l2) : 0.f;
                s_pt[c0 + j][r] = (__bf16)p;
                s_dst[c0 + j][r] = (__bf16)(p * (dp - dval));
            }
        }
        lds_barrier();
        // s_part is dead: scatter this wave's Q / dO A fragments transposed
        // (wave-private, no barrier)
#pragma unroll
        for (int mb = 0; mb < 2; ++mb)
#pragma unroll
            for (int c = 0; c < 4; ++c)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = 16 * c + lk + j, m = 16 * mb + lr;
                    s_qt[w][d][swz(d, m)] = qa[mb][c][j];
                    s_dot[w][d][swz(d, m)] = oa[mb][c][j];
                }
#pragma unroll
        for (int kc = 0; kc < 2; ++kc) {
            const bf16x8 pa = *reinterpret_cast<const bf16x8 *>(&s_pt[lr][16 * kc + lk]);
            const bf16x8 dsa = *reinterpret_cast<const bf16x8 *>(&s_dst[lr][16 * kc + lk]);
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int d = 16 * c + lr, m = 16 * kc + lk;
                const bf16x8 ob2 = *reinterpret_cast<const bf16x8 *>(&s_dot[w][d][swz(d, m)]);
                const bf16x8 qb2 = *reinterpret_cast<const bf16x8 *>(&s_qt[w][d][swz(d, m)]);
                dv[c] = wmma(pa, ob2, dv[c]);
                dk[c] = wmma(dsa, qb2, dk[c]);
            }
        }
        lds_barrier();
    }
    __bf16 *dkb = a.dk + b * a.dk_b + h * a.dk_h;
    __bf16 *dvb = a.dv + b * a.dv_b + h * a.dv_h;
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kj = n0 + lk + j;
            if (kj < Sk) {
                dkb[kj * a.dk_s + dcol0 + 16 * c + lr] = (__bf16)(dk[c][j] * a.scale);
                dvb[kj * a.dv_s + dcol0 + 16 * c + lr] = (__bf16)dv[c][j];
            }
        }
#endif
}

// ---------------------------------------------------------------- host
// launcher
inline hipError_t run(const Args &a, hipStream_t stream) {
    hipLaunchKernelGGL(dot_do_o_kernel, dim3((a.Sq + 3) / 4, a.H, a.B), dim3(128), 0, stream, a);
    if (const auto error = hipGetLastError(); error != hipSuccess)
        return error;
    hipLaunchKernelGGL(dq_kernel, dim3((a.Sq + QM0 - 1) / QM0, a.H, a.B), dim3(128), 0, stream, a);
    if (const auto error = hipGetLastError(); error != hipSuccess)
        return error;
    hipLaunchKernelGGL(dkdv_kernel, dim3((a.Sk + KN0 - 1) / KN0, a.H, a.B), dim3(128), 0, stream,
                       a);
    if (const auto error = hipGetLastError(); error != hipSuccess)
        return error;
    return hipSuccess;
}

} // namespace fa_d256
