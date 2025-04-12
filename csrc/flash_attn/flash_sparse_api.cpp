#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash_sparse.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

//
// Bit hacky but for now hook into the existing set_params_fprop, 
// set_params_splitkv, and set_params_alibi in flash_api.cpp
//
void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      const float softcap,
                      bool seqlenq_ngroups_swapped=false,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, const int num_sm, struct c10::TensorOptions opts) {

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (max_seqlen_q + 64 - 1) / 64;
    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
            params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}

void set_params_alibi(Flash_fwd_params &params, std::optional<at::Tensor> &alibi_slopes_, int batch_size, int num_heads){
#ifdef FLASHATTENTION_DISABLE_ALIBI
    TORCH_CHECK(!alibi_slopes_.has_value(), "This flash attention build does not support alibi.");
    params.alibi_slopes_ptr = nullptr;
#else
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({num_heads}) || alibi_slopes.sizes() == torch::IntArrayRef({batch_size, num_heads}));
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }
#endif
}

///////////////////////////////////////////////////////////////////////////////

void set_params_fprop_sparse(Flash_fwd_params_sparse &params,
                            // sizes
                            const size_t b,
                            const size_t seqlen_q,
                            const size_t seqlen_k,
                            const size_t seqlen_q_rounded,
                            const size_t seqlen_k_rounded,
                            const size_t h,
                            const size_t h_k,
                            const size_t d,
                            const size_t d_rounded,
                            // device pointers
                            const at::Tensor q,
                            const at::Tensor k,
                            const at::Tensor v,
                            const at::Tensor block_count,
                            const at::Tensor block_offset,
                            const at::Tensor column_count,
                            const at::Tensor column_index,
                            at::Tensor out,
                            void *cu_seqlens_q_d,
                            void *cu_seqlens_k_d,
                            void *seqused_k,
                            void *p_d,
                            void *softmax_lse_d,
                            float p_dropout,
                            float softmax_scale,
                            const float softcap,
                            bool seqlenq_ngroups_swapped=false,
                            const bool unpadded_lse=false) {
    set_params_fprop(params,
        b,
        seqlen_q, seqlen_k,
        seqlen_q_rounded, seqlen_k_rounded,
        h, h_k,
        d, d_rounded,
        q, k, v, out,
        cu_seqlens_q_d,
        cu_seqlens_k_d,
        seqused_k,
        p_d,
        softmax_lse_d,
        p_dropout,
        softmax_scale,
        -1,  // window_size_left
        -1,  // window_size_right
        softcap,
        seqlenq_ngroups_swapped,
        unpadded_lse
    );
    params.block_count = block_count.const_data_ptr<int>();
    params.block_offset = block_offset.const_data_ptr<int>();
    params.column_count = column_count.const_data_ptr<int>();
    params.column_index = column_index.const_data_ptr<int>();
    TORCH_CHECK(block_count.size(2) == block_offset.size(2));
    TORCH_CHECK(column_index.size(2) == block_offset.size(2));
    TORCH_CHECK(column_count.size(2) == column_index.size(2));
    params.NUM_ROWS = block_count.size(2);
    // params.NUM_ROWS should be equal to cdiv(seqlen_q, BLOCK_M), and BLOCK_M has to be 64 for now.
    constexpr int BLOCK_M = 64;
    int expected_num_rows = (seqlen_q + BLOCK_M - 1) / BLOCK_M;
    TORCH_CHECK(params.NUM_ROWS == expected_num_rows);
    params.NNZ_S = block_offset.size(3);
    params.NNZ_V = column_index.size(3);
}

void run_mha_fwd_sparse(Flash_fwd_params_sparse &params, cudaStream_t stream, bool force_split_kernel=false) {
    TORCH_CHECK(params.num_splits <= 1 && !force_split_kernel, "run_mha_fwd_sparse does not support splitkv.");
    TORCH_CHECK(params.d == 128, "run_mha_fwd_sparse only supports headdim=128 for now to keep binary small.");
    FP16_SWITCH(!params.is_bf16, [&] {
        constexpr static int kHeadDim = 128;
        BOOL_SWITCH(params.is_causal, Is_causal, [&] {
            run_mha_fwd_sparse_<elem_type, kHeadDim, Is_causal>(params, stream);
        });
    });
}

std::vector<at::Tensor>
mha_fwd_sparse(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
            const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
            const at::Tensor &block_count,
            const at::Tensor &block_offset,
            const at::Tensor &column_count,
            const at::Tensor &column_index,
            const std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
            const std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
            const double p_dropout,
            const double softmax_scale,
            bool is_causal,
            const double softcap,
            const bool return_softmax,
            std::optional<at::Generator> gen_) {

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    bool is_sm90 = cc_major == 9 && cc_minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    Flash_fwd_params_sparse params;
    set_params_fprop_sparse(params,
                            batch_size,
                            seqlen_q, seqlen_k,
                            seqlen_q_rounded, seqlen_k_rounded,
                            num_heads, num_heads_k,
                            head_size, head_size_rounded,
                            q_padded, k_padded, v_padded,
                            block_count, block_offset,
                            column_count, column_index,
                            out,
                            /*cu_seqlens_q_d=*/nullptr,
                            /*cu_seqlens_k_d=*/nullptr,
                            /*seqused_k=*/nullptr,
                            return_softmax ? p.data_ptr() : nullptr,
                            softmax_lse.data_ptr(),
                            p_dropout,
                            softmax_scale,
                            softcap
                    );

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen_k, seqlen_q,
        head_size_rounded, p_dropout, /*num_splits*/ 1, get_num_sm(get_current_device()), opts);

    // NOTE(woosuk): Commented out because they are not used in inference.
    // // number of times random will be generated per thread, to offset philox counter in thc random
    // // state
    // // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    // int64_t counter_offset = params.b * params.h * 32;
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // // Forward kernel will populate memory with the seed and offset.
    // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    // if (p_dropout > 0.0)  {
    //     auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //         gen_, at::cuda::detail::getDefaultCUDAGenerator());
    //     // See Note [Acquire lock when using random generators]
    //     std::lock_guard<std::mutex> lock(gen->mutex_);
    //     params.philox_args = gen->philox_cuda_state(counter_offset);
    // }

    // for alibi_slopes_ cast away constness that was added for torch library
    // compatibility, needs to be cast away to maintain compatibility with
    // upstream
    set_params_alibi(params, 
        const_cast<std::optional<at::Tensor> &>(alibi_slopes_), 
        batch_size, num_heads);

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_sparse(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, softmax_lse};
}

std::vector<at::Tensor>
mha_varlen_fwd_sparse(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                    const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                    const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i.
                    const at::Tensor &block_count,
                    const at::Tensor &block_offset,
                    const at::Tensor &column_count,
                    const at::Tensor &column_index,
                    const c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                    const at::Tensor &cu_seqlens_q,  // b+1
                    const at::Tensor &cu_seqlens_k,  // b+1
                    const c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
                    const c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
                    int64_t max_seqlen_q,
                    const int64_t max_seqlen_k,
                    const double p_dropout,
                    const double softmax_scale,
                    const bool zero_tensors,
                    bool is_causal,
                    const double softcap,
                    const bool return_softmax,
                    c10::optional<at::Generator> gen_) {

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor >= 0;
    bool is_sm90 = cc_major == 9 && cc_minor == 0;
    TORCH_CHECK(is_sm90 || is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");
    // We will support Turing in the near future
    // TORCH_CHECK(is_sm90 || is_sm8x || is_sm75, "FlashAttention only supports Turing GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    if (q_dtype == torch::kBFloat16) {
        TORCH_CHECK(is_sm90 || is_sm8x, "bfloat16 is only supported on Ampere GPUs or newer");
    }
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int num_heads_k = k.size(1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const int total_q = q.sizes()[0];

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    const int total_k = k.size(0);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);


    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, sizes[0], sizes[1], head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        p = torch::empty({ batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params_sparse params;
    set_params_fprop_sparse(params,
                            batch_size,
                            max_seqlen_q, max_seqlen_k,
                            seqlen_q_rounded, seqlen_k_rounded,
                            num_heads, num_heads_k,
                            head_size, head_size_rounded,
                            q_padded, k_padded, v_padded,
                            block_count, block_offset,
                            column_count, column_index,
                            out,
                            cu_seqlens_q_d,
                            cu_seqlens_k.data_ptr(),
                            seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                            return_softmax ? p.data_ptr() : nullptr,
                            softmax_lse.data_ptr(),
                            p_dropout,
                            softmax_scale,
                            softcap
                    );
    params.total_q = total_q;

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;

    // NOTE(woosuk): Commented out because they are not used in inference.
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    // int64_t counter_offset = params.b * params.h * 32;
    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    // // Forward kernel will populate memory with the seed and offset.
    // params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    // if (p_dropout > 0.0)  {
    //     auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
    //         gen_, at::cuda::detail::getDefaultCUDAGenerator());
    //     // See Note [Acquire lock when using random generators]
    //     std::lock_guard<std::mutex> lock(gen->mutex_);
    //     params.philox_args = gen->philox_cuda_state(counter_offset);
    // }

    // for alibi_slopes_ cast away constness that was added for torch library
    // compatibility, needs to be cast away to maintain compatibility with
    // upstream
    set_params_alibi(params, 
        const_cast<std::optional<at::Tensor> &>(alibi_slopes_), 
        batch_size, num_heads);

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd_sparse(params, stream);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // at::Tensor out_padded = out;
    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
    }

    return {out, softmax_lse};
}

} // namespace FLASH_NAMESPACE