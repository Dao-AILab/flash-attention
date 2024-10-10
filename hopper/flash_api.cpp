/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t b_k,
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
                      void *seqused_q,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool seqlenq_ngroups_swapped=false,
                      bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;
    params.is_kv_cache = false;
    params.page_num_blocks = 0;
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
    params.seqused_q = static_cast<int *>(seqused_q);
    params.seqused_k = static_cast<int *>(seqused_k);

    TORCH_CHECK(
        bool(params.cu_seqlens_q) == bool(params.cu_seqlens_k),
        "cu_seqlens_q and cu_seqlens_k must be both null or non-null"
    );

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.b_k = b_k;
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
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

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
    window_size_left = std::min(int(seqlen_k), window_size_left);
    window_size_right = std::min(int(seqlen_k), window_size_right);
    if (window_size_left < 0) { window_size_left = seqlen_k; }
    if (window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_causal = window_size_left == int(seqlen_k) && window_size_right == 0;
    if ((window_size_left < int(seqlen_k) || window_size_right < int(seqlen_k)) && !params.is_causal) {
        params.is_local = true;
    }

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
    params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
}

void set_params_dgrad(Flash_bwd_params &params,
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
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_q,
                      void *seqused_k,
                      void *dq_accum_d,
                      void *dk_accum_d,
                      void *dv_accum_d,
                      void *softmax_lse_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool deterministic) {

    set_params_fprop(params,
                     b, b, seqlen_q, seqlen_k, seqlen_q_rounded, seqlen_k_rounded, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k_d,
                     seqused_q,
                     seqused_k,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.page_num_blocks = 0;
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);

    if (cu_seqlens_q_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;
    params.dk_accum_ptr = dk_accum_d;
    params.dv_accum_ptr = dv_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;

    params.deterministic = deterministic;
}


// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 80%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks, int batch_nheads, int num_SMs, int num_n_blocks,
    int max_splits, int head_size, bool use_one_mma_wg) {
    // Goal of the starting threshold is to determine whether to split or not.
    // Empirically, the efficiency threshold can be much lower than 80% depending on num_n_blocks.
    int num_m_blocks = batch_nheads_mblocks/batch_nheads;
    float start_threshold;
    float num_n_blocksf = float(num_n_blocks);
    if (head_size == 128) {
        if (std::log2f(num_n_blocksf) <= 4) { // 2048 -- .25
            start_threshold = .20f + (std::log2f(num_n_blocksf) - 3) * .05f;
        } else if (std::log2f(num_n_blocksf) <= 5) { // 4096 -- .25
            start_threshold = .25f;
        } else if (std::log2f(num_n_blocksf) <= 6) { // 8192 -- .36
            start_threshold = .28f + (std::log2f(num_n_blocksf) - 5) * .08f;
        } else if (std::log2f(num_n_blocksf) <= 7) { // 16K -- .42
            start_threshold = .36f + (std::log2f(num_n_blocksf) - 6) * .06f;
        } else {
            // Just split freely
            start_threshold = .8f;
        }
        if (num_m_blocks > 1 && start_threshold < .5f)
            start_threshold += .05f * (std::log2f(num_n_blocksf) - 2);
    } else if (head_size == 256) {
        // TODO for hdim 256
        if (num_n_blocks <= 40) {
            start_threshold = .24f;
        } else if (std::log2f(num_n_blocksf) <= 8) {
            start_threshold = .33f + std::max(0.f, (std::log2f(num_n_blocksf) - std::log2f(50)) * 0.02971f);
        } else {
            // Just split freely
            start_threshold = .8f;
        }
    } else if (head_size == 64) {
        if (use_one_mma_wg) {
            if (std::log2f(num_n_blocksf) <= 4) { // 2K -- .33
                start_threshold = .33f;
            } else if (std::log2f(num_n_blocksf) <= 5) { // 4K -- .37
                start_threshold = .33f + (std::log2f(num_n_blocksf) - 4) * .04f;
            } else if (std::log2f(num_n_blocksf) <= 6) { // 8K -- .40
                start_threshold = .37f + (std::log2f(num_n_blocksf) - 5) * .03f;
            } else if (std::log2f(num_n_blocksf) <= 7) { // 16K -- .43
                start_threshold = .4f + (std::log2f(num_n_blocksf) - 6) * .03f;
            } else if (std::log2f(num_n_blocksf) <= 8) { // 32K -- .46
                start_threshold = .43f + (std::log2f(num_n_blocksf) - 7) * .03f;
            } else {
                start_threshold = .8f;
            }
        } else {
            if (std::log2f(num_n_blocksf) <= 6) { // 8K -- .5
                start_threshold = .5f;
            } else {
                start_threshold = .8f;
            }
        }
    } else {
        // placeholder for other hdims
        start_threshold = .8f;
    }

    float first_wave = float(batch_nheads_mblocks) / num_SMs;
    // printf("Start threshold and wave = %f, %f.\n", start_threshold, first_wave);
    // Only use start_threshold if initial work doesn't exceed one wave
    if ((first_wave/ceil(first_wave) > start_threshold && first_wave <= 1.f) ||
        (first_wave/ceil(first_wave) > .8f)) {
        return 1;
    }
    // if (first_wave_batch_nheads > start_threshold) { return 1; }
    // if (first_wave_batch_nheads > start_threshold || first_wave > .8f) { return 1; }
    // if (float(batch_nheads)/num_SMs > start_threshold) { return 1; }

    // If num_n_blocks is too small, use 1 split
    // For example, we never split for hdim = 128 and seqlen_k = 512,
    // or for hdim = 128, seqlen_k = 1024, and one MMA warpgroup.
    if (num_n_blocks < 8 || (use_one_mma_wg && num_n_blocks < 10)) { return 1; }

    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    
    // NOTE: disable split eligibility check for FA3 since we have dynamic tile scheduler
    // for exiting splits with no work early, and check leads to efficiency quantization issues.
    // Comment from FA2:
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    // auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    //     return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    // };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        // if (!is_split_eligible(num_splits)) {
        //     efficiency.push_back(0.f);
        // } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            // printf("num_splits = %d, n_waves = %f, ceil(n_waves) = %f,  eff = %f\n", num_splits, n_waves, ceil(n_waves), eff);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        // }
    }
    // Correct for excessive splitting with e.g. 1 bsz*nheads*mblocks
    // Empirically, efficiency threshold in these cases is about 40% for 64K seqlen_k
    float threshold = num_m_blocks == 1 ? std::min(0.3f + batch_nheads * 0.1f, 0.8f) : 0.8f;
    threshold = threshold * max_efficiency;
    // printf("Max efficiency = %f. Threshold = %f.\n", max_efficiency, threshold);
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        // if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] > threshold) {
            // printf("num_splits chosen = %d, threshold = %f, efficiency = %f.\n", num_splits, threshold, efficiency[num_splits - 1]);
            return num_splits;
        }
    }
    return 1;
}

std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int num_heads_k, const int head_size, const int max_seqlen_k, const int max_seqlen_q,
    const int head_size_rounded, const float p_dropout,
    const int num_splits, cudaDeviceProp *dprops, bool use_gqa_packing, bool is_causal, struct c10::TensorOptions opts) {
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

    params.num_splits = num_splits;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        if (num_splits < 1) {
            const int gqa_ratio = num_heads / num_heads_k;
            const int block_h = 1 << static_cast<int>(std::ceil(std::log2(std::clamp(gqa_ratio, 1, 32))));
            const int block_m = head_size == 64 ? 192 : 128;
            const bool use_one_mma_wg = max_seqlen_q <= 64/block_h;
            
            int block_n = 128;
            if (head_size == 128 && !is_causal) {
                block_n = 176;
            } else if (head_size == 256) {
                block_n = use_one_mma_wg ? 96 : 80;
            }
            const int num_n_blocks = (max_seqlen_k + block_n - 1) / block_n;
            const int batch_nheads = use_gqa_packing ? batch_size * num_heads_k : batch_size * num_heads;
            const int batch_nheads_mblocks = use_gqa_packing
                ? ceildiv(max_seqlen_q, block_m / block_h) * batch_nheads
                : ceildiv(max_seqlen_q, block_m) * batch_nheads;
            params.num_splits = num_splits_heuristic(batch_nheads_mblocks, batch_nheads,
                dprops->multiProcessorCount, num_n_blocks, 128, head_size, use_one_mma_wg);
            // printf("Num splits heuristic = %d.\n", params.num_splits);
	    }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, max_seqlen_q, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
            params.oaccum_row_stride = out_accum.stride(-2);
            params.oaccum_head_stride = out_accum.stride(-3);
            params.oaccum_batch_stride = out_accum.stride(-4);
            params.oaccum_split_stride = out_accum.stride(0);
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    return std::make_tuple(softmax_lse_accum, out_accum);
}


void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream, bool force_split_kernel=false) { 

    int dtype = 1;
    if (params.is_bf16) { dtype = 2; }
    else if (params.is_e4m3) { dtype = 3; }
    PREC_SWITCH(dtype, Element, [&] {
      HEADDIM_SWITCH(params.d, kHeadSize, [&] {
        if(!params.use_gqa_packing) {
          run_mha_fwd_<Element, kHeadSize>(params, stream);
        } else {
          QUERYHEAD_SWITCH(params.h_h_k_ratio, kBlockH, [&] {
            run_mha_fwd_gqa_<Element, kHeadSize, kBlockH>(params, stream);
          });
        }
      });
    });

#if 0
    if (!params.is_e4m3) { 
        if (params.is_bf16) {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::bfloat16_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::bfloat16_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::bfloat16_t, 256>(params, stream);
            }
        } else {
            if (params.d == 64) {
                run_mha_fwd_<cutlass::half_t, 64>(params, stream);
            } else if (params.d == 128) {
                run_mha_fwd_<cutlass::half_t, 128>(params, stream);
            } else {
                run_mha_fwd_<cutlass::half_t, 256>(params, stream);
            }
        }
    } else {
        if (params.d == 64) {
            run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
        } else if (params.d == 128) {
            run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
        } else if (params.d == 256) {
            run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
        }
    }
#endif
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        c10::optional<at::Tensor> &descale_q_, // 1
        c10::optional<at::Tensor> &descale_k_, // 1
        c10::optional<at::Tensor> &descale_v_, // 1
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool use_gqa_packing = false
        ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention-3 only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
                "FlashAttention-3 only support fp16, bf16, or fp8 e4m3 data type");
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
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // Guard against mistaken setting of gqa flag
    if (num_heads == num_heads_k) { use_gqa_packing = false; }

    TORCH_CHECK(head_size_og == 64 || head_size_og == 128 || head_size_og == 256, "Only support head size 64, 128, and 256 for now");

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
        // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn
                    ? (out.dtype() == at::kBFloat16)
                    : (out.dtype() == q_dtype),
                "Output must have the same dtype as input dtype if dtype is "
                "not fp8, or fp16 for fp8 input.");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (q_dtype == at::ScalarType::Float8_e4m3fn)
            out = torch::empty_like(q_padded, at::kBFloat16);
        else
            out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right);

    auto tile_count_semaphore = is_causal || params.is_local
        ? torch::zeros({1}, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    at::Tensor descale_q, descale_k, descale_v;
    if(q_dtype == at::ScalarType::Float8_e4m3fn) {
        if (descale_q_.has_value()) {
            descale_q = descale_q_.value();
            CHECK_DEVICE(descale_q);
            CHECK_SHAPE(descale_q, 1);
        } else { descale_q = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_k_.has_value()) {
            descale_k = descale_k_.value();
            CHECK_DEVICE(descale_k);
            CHECK_SHAPE(descale_k, 1);
        } else { descale_k = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_v_.has_value()) {
            descale_v = descale_v_.value();
            CHECK_DEVICE(descale_v);
            CHECK_SHAPE(descale_v, 1);
        } else { descale_v = torch::ones({1}, opts.dtype(at::kFloat)); }
        params.descale_q_ptr = descale_q.data_ptr<float>();
        params.descale_k_ptr = descale_k.data_ptr<float>();
        params.descale_v_ptr = descale_v.data_ptr<float>();
    } else {
        params.descale_q_ptr = nullptr;
        params.descale_k_ptr = nullptr;
        params.descale_v_ptr = nullptr;
    }
    
    params.use_gqa_packing = use_gqa_packing;

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
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

    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
               int max_seqlen_q,
               const int max_seqlen_k,
               const float softmax_scale,
               bool is_causal,
               int window_size_left,
               int window_size_right) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    void *cu_seqlens_q_d = cu_seqlens_q.data_ptr();

    const int total_q = q.sizes()[0];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? -1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value  must divide number of heads in query");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    const int total_k = k.size(0);

    if (!paged_KV) {
        CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    if (seqused_q.has_value()){
        auto seqused_q_ = seqused_q.value();
        TORCH_CHECK(seqused_q_.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q_.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q_.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q_, batch_size);
    }

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

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, k_padded, v_padded, out,
                     cu_seqlens_q_d,
                     cu_seqlens_k.data_ptr(),
                     seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     /*p_d=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     /*seqlenq_ngroups_swapped=*/false,
                     /*unpadded_lse=*/true);
    params.total_q = total_q;
    params.total_k = total_k;

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.page_num_blocks = k.size(0);
    }
    params.page_block_size = page_block_size;
    params.page_num_blocks = num_blocks;

    //printf("mha_varlen_fwd: params.seqlen_k=%d, max_seqlen_k=%d, params.page_num_blocks=%d\n", (int)params.seqlen_k, (int)max_seqlen_k, (int)params.page_num_blocks);
    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
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
    return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
  // FP16_SWITCH(!params.is_bf16, [&] {
  //     HEADDIM_SWITCH(params.d, [&] {
  //         run_mha_bwd_<elem_type, kHeadDim>(params, stream);
  //     });
  // });
  if (!params.is_bf16) {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::half_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::half_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::half_t, 128>(params, stream);
    }
  } else {
    if (params.d <= 64) {
      run_mha_bwd_<cutlass::bfloat16_t, 64>(params, stream);
    } else if (params.d <= 96) {
      run_mha_bwd_<cutlass::bfloat16_t, 96>(params, stream);
    } else {
      run_mha_bwd_<cutlass::bfloat16_t, 128>(params, stream);
    }
  }
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen_q = sizes[1];
    const int num_heads = sizes[2];
    const int head_size_og = dout.size(3);
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : round_multiple(head_size, 32);
    // This should match the kernel configs
    const int kBlockM = head_size <= 64 ? 128 : (head_size < 256 ? 64 : 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size_og);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have seqlen_q_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({batch_size, num_heads, seqlen_q_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    dq_accum = torch::empty({batch_size, num_heads, seqlen_q_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    // dk_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));
    // dv_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if (is_causal) { window_size_right = 0; }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     dq_accum.data_ptr(),
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right,
                     deterministic);
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();
    // printf("dq_semaphore: %p, [%d, %d, %d]\n", params.dq_semaphore, (seqlen_q + 64 - 1) / 64, batch_size, num_heads);

    if (seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum};
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
               const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
               const at::Tensor &softmax_lse,     // b x h x seqlen_q
               c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
               c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
               c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               c10::optional<at::Tensor> &seqused_q, // b. If given, only this many elements of each batch element's queries and outputs are used.
               c10::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
               const int max_seqlen_q,
               const int max_seqlen_k,          // max sequence length to choose the kernel
               const float softmax_scale,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This flash attention build does not support backward.");
    #endif
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm9x = dprops->major == 9 && dprops->minor >= 0;
    TORCH_CHECK(is_sm9x, "FlashAttentionHopper only supports Hopper GPUs or newer.");

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_og = dout.size(2);
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 128, "FlashAttention backward only supports head dimension at most 128");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = head_size <= 64 ? 64 : round_multiple(head_size, 32);
    // This should match the kernel configs
    const int kBlockM = head_size <= 64 ? 128 : (head_size < 256 ? 64 : 32);
    const int seqlen_q_rounded = round_multiple(max_seqlen_q, kBlockM);
    const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);
    int const total_q_padded_rounded = round_multiple(total_q + batch_size * 128, 128);

    TORCH_CHECK(head_size == round_multiple(head_size_og, 8), "head_size must be head_size_og rounded to a multiple of 8");

    CHECK_SHAPE(q, total_q, num_heads, head_size_og);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_og);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_og);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    if (seqused_q.has_value()){
        auto seqused_q_ = seqused_q.value();
        TORCH_CHECK(seqused_q_.dtype() == torch::kInt32, "seqused_q must have dtype int32");
        TORCH_CHECK(seqused_q_.is_cuda(), "seqused_q must be on CUDA device");
        TORCH_CHECK(seqused_q_.is_contiguous(), "seqused_q must be contiguous");
        CHECK_SHAPE(seqused_q_, batch_size);
    }

    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    if (seqused_k.has_value()){
        auto seqused_k_ = seqused_k.value();
        TORCH_CHECK(seqused_k_.dtype() == torch::kInt32, "seqused_k must have dtype int32");
        TORCH_CHECK(seqused_k_.is_cuda(), "seqused_k must be on CUDA device");
        TORCH_CHECK(seqused_k_.is_contiguous(), "seqused_k must be contiguous");
        CHECK_SHAPE(seqused_k_, batch_size);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size);
    } else {
        dv = torch::empty_like(v);
    }

    at::Tensor dout_padded;
    if (head_size_og % 8 != 0) {
        dout_padded = torch::nn::functional::pad(dout, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        dout_padded = dout;
    }

    if (is_causal) { window_size_right = 0; }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();
    // Need softmax_d to have total_q_padded_rounded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    auto softmax_d = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    auto softmax_lse_log2 = torch::empty({num_heads, total_q_padded_rounded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    at::Tensor dk_accum, dv_accum;
    dq_accum = torch::empty({num_heads, total_q_padded_rounded, head_size_rounded}, opts.dtype(at::kFloat));
    // dk_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));
    // dv_accum = torch::zeros({batch_size, seqlen_k_rounded, num_heads_k, head_size_rounded}, opts.dtype(at::kFloat));

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     max_seqlen_q, max_seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     dout_padded, dq, dk_expanded, dv_expanded,
                     cu_seqlens_q.data_ptr(),
                     cu_seqlens_k.data_ptr(),
                     seqused_q.has_value() ? seqused_q.value().data_ptr() : nullptr,
                     seqused_k.has_value() ? seqused_k.value().data_ptr() : nullptr,
                     dq_accum.data_ptr(),
                     // loop ? dk_accum.data_ptr() : nullptr,
                     // loop ? dv_accum.data_ptr() : nullptr,
                     nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     softmax_d.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/window_size_left,
                     /*window_size_right=*/window_size_right,
                     deterministic);
    params.total_q = total_q;
    params.total_k = total_k;
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();

    // Will be zero'ed out in the backward preprocess kernel
    at::Tensor dq_semaphore = torch::empty({(max_seqlen_q + kBlockM - 1) / kBlockM, batch_size, num_heads}, opts.dtype(torch::kInt32));
    params.dq_semaphore = dq_semaphore.data_ptr<int>();

    if (max_seqlen_q > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If max_seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    if (head_size_og % 8 != 0) {
        dq = dq.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dk = dk.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        dv = dv.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    }

    return { dq, dk, dv, softmax_d, dq_accum, softmax_lse_log2 };
}

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,                 // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,            // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                c10::optional<const at::Tensor> &k_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &v_, // batch_size x seqlen_knew x num_heads_k x head_size
                c10::optional<const at::Tensor> &seqlens_k_, // batch_size
                c10::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
                c10::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                c10::optional<const at::Tensor> &leftpad_k_, // batch_size
                c10::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
                c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
                c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                c10::optional<at::Tensor> &descale_q_, // 1
                c10::optional<at::Tensor> &descale_k_, // 1
                c10::optional<at::Tensor> &descale_v_, // 1
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float softcap,
                bool is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits,
                int max_seqlen_k_hint,
                bool use_gqa_packing
                ) {

    auto dprops = at::cuda::getCurrentDeviceProperties();
    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    // bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
    TORCH_CHECK(is_sm90, "FlashAttention-3 only supports Hopper GPUs or newer.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16 || q_dtype == at::ScalarType::Float8_e4m3fn,
                "FlashAttention-3 only support fp16, bf16, or fp8 e4m3 data type");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(kcache); CHECK_DEVICE(vcache);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        TORCH_CHECK(!cache_batch_idx_.has_value(), "Paged KVcache does not support cache_batch_idx");
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size_og = sizes[3];

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 1 : kcache.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 256 == 0, "Paged KV cache block size must be divisible by 256");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // Guard against mistaken setting of gqa flag
    if (num_heads == num_heads_k) { use_gqa_packing = false; }

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped =
        seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 &&
        window_size_right < 0 && head_size_og % 8 == 0 &&
        !alibi_slopes_.has_value() && !use_gqa_packing;
    if (seqlenq_ngroups_swapped) {
        const int ngroups = num_heads / num_heads_k;
        q = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_heads_k;
    }

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    if (!paged_KV) {
        CHECK_SHAPE(kcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, batch_size_c, seqlen_k, num_heads_k, head_size_og);
    } else {
        CHECK_SHAPE(kcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(vcache, num_blocks, page_block_size, num_heads_k, head_size_og);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    at::Tensor q_padded, kcache_padded, vcache_padded;
    if (head_size_og % 8 != 0) {
        q_padded = torch::nn::functional::pad(q, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        kcache_padded = torch::nn::functional::pad(kcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        vcache_padded = torch::nn::functional::pad(vcache, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
    } else {
        q_padded = q;
        kcache_padded = kcache;
        vcache_padded = vcache;
    }

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        // TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        TORCH_CHECK(q_dtype == at::ScalarType::Float8_e4m3fn
                    ? (out.dtype() == at::kBFloat16)
                    : (out.dtype() == q_dtype),
                "Output must have the same dtype as input dtype if dtype is "
                "not fp8, or fp16 for fp8 input.");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        if (q_dtype == at::ScalarType::Float8_e4m3fn) {
            out = torch::empty_like(q_padded, at::kBFloat16);
        }
        else
            out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size = round_multiple(head_size_og, 8);
    const int head_size_rounded = head_size <= 192 ? round_multiple(head_size, 32) : 256;
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, batch_size_c,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q_padded, kcache_padded, vcache_padded, out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_q=*/nullptr,
                     /*seqused_k=*/nullptr,
                     /*p_ptr=*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right
                     );

    at::Tensor descale_q, descale_k, descale_v;
    if(q_dtype == at::ScalarType::Float8_e4m3fn) {
        if (descale_q_.has_value()) {
            descale_q = descale_q_.value();
            CHECK_DEVICE(descale_q);
            CHECK_SHAPE(descale_q, 1);
        } else { descale_q = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_k_.has_value()) {
            descale_k = descale_k_.value();
            CHECK_DEVICE(descale_k);
            CHECK_SHAPE(descale_k, 1);
        } else { descale_k = torch::ones({1}, opts.dtype(at::kFloat)); }
        if (descale_v_.has_value()) {
            descale_v = descale_v_.value();
            CHECK_DEVICE(descale_v);
            CHECK_SHAPE(descale_v, 1);
        } else { descale_v = torch::ones({1}, opts.dtype(at::kFloat)); }
        params.descale_q_ptr = descale_q.data_ptr<float>();
        params.descale_k_ptr = descale_k.data_ptr<float>();
        params.descale_v_ptr = descale_v.data_ptr<float>();
    } else {
        params.descale_q_ptr = nullptr;
        params.descale_k_ptr = nullptr;
        params.descale_v_ptr = nullptr;
    }
    
    params.is_kv_cache = true;

    params.use_gqa_packing = use_gqa_packing;

    at::Tensor k, v, k_padded, v_padded;
    if (k_.has_value()) {
        TORCH_CHECK(v_.has_value(), "If key is supplied, value must also be passed in");
        TORCH_CHECK(seqlens_k_.has_value(), "If key is supplied, seqlens_k must also be passed in");
        TORCH_CHECK(seqlen_q <= seqlen_k, "If key is supplied, it must have seqlen <= the seqlen of the KV cache");
        k = k_.value();
        v = v_.value();
        TORCH_CHECK(k.dtype() == q_dtype, "Key must have the same dtype as query");
        TORCH_CHECK(v.dtype() == q_dtype, "Value must have the same dtype as query");
        CHECK_DEVICE(k); CHECK_DEVICE(v);
        TORCH_CHECK(k.stride(-1) == 1, "Key tensor must have contiguous last dimension");
        TORCH_CHECK(v.stride(-1) == 1, "Value tensor must have contiguous last dimension");
        int seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
        params.seqlen_knew = seqlen_knew;
        params.knew_ptr = k_padded.data_ptr();
        params.vnew_ptr = v_padded.data_ptr();
        // All stride are in elements, not bytes.
        params.knew_batch_stride = k_padded.stride(0);
        params.vnew_batch_stride = v_padded.stride(0);
        params.knew_row_stride = k_padded.stride(-3);
        params.vnew_row_stride = v_padded.stride(-3);
        params.knew_head_stride = k_padded.stride(-2);
        params.vnew_head_stride = v_padded.stride(-2);
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);
        params.seqused_k = static_cast<int *>(seqlens_k.data_ptr());
    }
    if (leftpad_k_.has_value()) {
        TORCH_CHECK(!paged_KV, "We don't support Paged KV and leftpad_k running at the same time yet");
        auto leftpad_k = leftpad_k_.value();
        TORCH_CHECK(leftpad_k.dtype() == torch::kInt32, "leftpad_k must have dtype int32");
        CHECK_DEVICE(leftpad_k);
        CHECK_CONTIGUOUS(leftpad_k);
        CHECK_SHAPE(leftpad_k, batch_size);
        TORCH_CHECK(false, "Left Padding K is not supported");
        //params.leftpad_k = static_cast<int *>(leftpad_k.data_ptr());
    }

    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        params.rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(params.rotary_dim <= head_size, "rotary_dim must be <= headdim");
        TORCH_CHECK(params.rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, params.rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
        params.rotary_cos_ptr = rotary_cos.data_ptr();
        params.rotary_sin_ptr = rotary_sin.data_ptr();
        params.is_rotary_interleaved = is_rotary_interleaved;
    } else {
        params.rotary_dim = 0;
    }

    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
        params.cache_batch_idx = reinterpret_cast<int *>(cache_batch_idx.data_ptr());
    }

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
       params, batch_size, num_heads, num_heads_k, head_size, max_seqlen_k_hint, seqlen_q,
       head_size_rounded, /*dropout*/ 0.f, num_splits, dprops, use_gqa_packing, is_causal, opts);
    
    auto tile_count_semaphore = is_causal || params.is_local || params.num_splits != 1
        ? torch::zeros({1}, opts.dtype(torch::kInt32))
        : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (paged_KV) {
        params.block_table = block_table.data_ptr<int>();
        params.block_table_batch_stride = block_table.stride(0);
    }
    params.page_block_size = page_block_size;

    TORCH_CHECK(!alibi_slopes_.has_value(), "Alibi Slopes are not supported yet");
    //set_params_alibi(params, alibi_slopes_, batch_size, num_heads);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Only split kernel supports appending to KV cache, or indexing to the cache with cache_batch_idx,
    // or paged KV cache
    //run_mha_fwd(params, stream, /*force_split_kernel=*/k_.has_value() || cache_batch_idx_.has_value() || paged_KV);
    run_mha_fwd(params, stream);

    if (head_size_og % 8 != 0) {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value()) { out_.value().copy_(out); }
        if (k_.has_value()) {
            // It's expensive to copy the KV cache here for the case where head size not divisible by 8,
            // but we don't expect to get this case in practice. This is just so that the code works for that case.
            kcache.copy_(kcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
            vcache.copy_(vcache_padded.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)}));
        }
    }

    if (seqlenq_ngroups_swapped) {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }

    return {out, softmax_lse};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("varlen_bwd", &mha_varlen_bwd, "Varlen backward pass");
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
}
