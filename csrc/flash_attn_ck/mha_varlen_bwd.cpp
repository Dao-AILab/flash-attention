/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "ck_build_config.hpp"
#include "fmha_bwd.hpp"
#include "mask.hpp"
#include <variant>

fmha_bwd_traits get_ck_fmha_varlen_bwd_traits(const mask_info &mask,
                                              std::string dtype,
                                              int head_size,
                                              bool has_dropout,
                                              bool enable_alibi,
                                              bool deterministic)
{
#if FLASHATTN_CK_USE_CURRENT_API
    return fmha_bwd_traits{
        .seqlen_q = -1,
        .seqlen_k = -1,
        .batch = -1,
        .max_seqlen_q = -1,
        .max_seqlen_k = -1,
        .hdim_q = head_size,
        .hdim_v = head_size,
        .nhead_q = -1,
        .nhead_k = -1,
        .data_type = std::move(dtype),
        .is_group_mode = true,
        .mask_type = mask.type,
        .bias_type = enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
        .has_dbias = false,
        .has_dropout = has_dropout,
        .is_store_randval = false,
        .is_deterministic = deterministic,
    };
#else
    return fmha_bwd_traits{head_size,
                           head_size,
                           dtype,
                           true,
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           false,
                           has_dropout,
                           false,
                           deterministic};
#endif
}

fmha_bwd_args get_ck_fmha_varlen_bwd_args(const mask_info &mask,
                                          // sizes
                                          const int b,
                                          const int max_seqlen_q,
                                          const int max_seqlen_k,
                                          const int h,
                                          const int h_k,
                                          const int hdim,
                                          // device pointers
                                          const at::Tensor q,
                                          const at::Tensor k,
                                          const at::Tensor v,
                                          const at::Tensor seqlens_q,
                                          const at::Tensor seqlens_k,
                                          std::optional<at::Tensor> &alibi_slopes_,
                                          const at::Tensor out,
                                          const at::Tensor softmax_lse,
                                          const at::Tensor dout,
                                          at::Tensor dq_acc,
                                          at::Tensor d,
                                          at::Tensor dq,
                                          at::Tensor dk,
                                          at::Tensor dv,
                                          float softmax_scale,
                                          float p_dropout,
                                          std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    ck_tile::index_t total_q = q.size(0);
    ck_tile::index_t total_k = k.size(0);

    // q: (total_q, nheads, hdim)
    ck_tile::index_t batch_stride_q = 0;
    ck_tile::index_t stride_q = q.stride(0);
    ck_tile::index_t nhead_stride_q = q.stride(1);

    // k: (total_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_k = 0;
    ck_tile::index_t stride_k = k.stride(0);
    ck_tile::index_t nhead_stride_k = k.stride(1);

    // v: (total_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_v = 0;
    ck_tile::index_t stride_v = v.stride(0);
    ck_tile::index_t nhead_stride_v = v.stride(1);

    // o: (total_q, nheads, hdim)
    ck_tile::index_t batch_stride_o = 0;
    ck_tile::index_t stride_o = out.stride(0);
    ck_tile::index_t nhead_stride_o = out.stride(1);

    // lse: (nheads, total_q)
    ck_tile::index_t batch_stride_lse = 0;
    ck_tile::index_t nhead_stride_lse = softmax_lse.stride(0);

    // do: (total_q, nheads, hdim)
    ck_tile::index_t batch_stride_do = 0;
    ck_tile::index_t stride_do = dout.stride(0);
    ck_tile::index_t nhead_stride_do = dout.stride(1);

    // d: (batch_size, nheads, max_seqlen_q)
    // CK assume d share the same stride with lse

    // dq: (total_q, nheads, hdim)
    ck_tile::index_t batch_stride_dq = 0;
    ck_tile::index_t stride_dq = dq.stride(0);
    ck_tile::index_t nhead_stride_dq = dq.stride(1);


    // dk_expanded: (total_k, nheads, hdim)
    ck_tile::index_t batch_stride_dk = 0;
    ck_tile::index_t stride_dk = dk.stride(0);
    ck_tile::index_t nhead_stride_dk = dk.stride(1);

    // dv_expanded: (total_k, nheads, hdim)
    ck_tile::index_t batch_stride_dv = 0;
    ck_tile::index_t stride_dv = dv.stride(0);
    ck_tile::index_t nhead_stride_dv = dv.stride(1);

    // dq_acc: (nheads, split, total_q, hdim)
    ck_tile::index_t split_stride_dq_acc = dq_acc.stride(1);
    ck_tile::index_t batch_stride_dq_acc = 0;
    ck_tile::index_t stride_dq_acc = dq_acc.stride(2);
    ck_tile::index_t nhead_stride_dq_acc = dq_acc.stride(0);

    float p_undrop = 1.0 - p_dropout;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        // alibi_slopes:(batch_size, nheads) or (nhead)
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

#if FLASHATTN_CK_USE_CURRENT_API
    auto drop_seed_var = std::variant<std::pair<uint64_t, uint64_t>,
                                      std::pair<const void*, const void*>>{
        std::pair<uint64_t, uint64_t>{0, 0}};
    if (drop_seed_offset.first != nullptr && drop_seed_offset.second != nullptr) {
        drop_seed_var = std::pair<const void*, const void*>{drop_seed_offset.first, drop_seed_offset.second};
    }
    return fmha_bwd_args{
        .q_ptr = q.data_ptr(),
        .k_ptr = k.data_ptr(),
        .v_ptr = v.data_ptr(),
        .bias_ptr = alibi_slopes_ptr,
        .o_ptr = out.data_ptr(),
        .lse_ptr = softmax_lse.data_ptr(),
        .do_ptr = dout.data_ptr(),
        .d_ptr = d.data_ptr(),
        .rand_val_ptr = nullptr,
        .dq_ptr = dq.data_ptr(),
        .dk_ptr = dk.data_ptr(),
        .dv_ptr = dv.data_ptr(),
        .dbias_ptr = nullptr,
        .dq_acc_ptr = dq_acc.data_ptr(),
        .seqstart_q_ptr = seqlens_q.data_ptr(),
        .seqstart_k_ptr = seqlens_k.data_ptr(),
        .seqlen_q_ptr = nullptr,
        .seqlen_k_ptr = nullptr,
        .cu_seqlen_q_ptr = nullptr,
        .cu_seqlen_k_ptr = nullptr,
        .seqlen_q = total_q,
        .seqlen_k = total_k,
        .batch = b,
        .max_seqlen_q = max_seqlen_q,
        .max_seqlen_k = max_seqlen_k,
        .hdim_q = hdim,
        .hdim_v = hdim,
        .nhead_q = h,
        .nhead_k = h_k,
        .scale = softmax_scale,
        .stride_q = stride_q,
        .stride_k = stride_k,
        .stride_v = stride_v,
        .stride_bias = stride_alibi_slopes,
        .stride_o = stride_o,
        .stride_randval = 0,
        .stride_do = stride_do,
        .stride_dq_acc = stride_dq_acc,
        .stride_dq = stride_dq,
        .stride_dk = stride_dk,
        .stride_dv = stride_dv,
        .stride_dbias = 0,
        .nhead_stride_q = nhead_stride_q,
        .nhead_stride_k = nhead_stride_k,
        .nhead_stride_v = nhead_stride_v,
        .nhead_stride_bias = 0,
        .nhead_stride_o = nhead_stride_o,
        .nhead_stride_randval = 0,
        .nhead_stride_do = nhead_stride_do,
        .nhead_stride_lsed = nhead_stride_lse,
        .nhead_stride_dq_acc = nhead_stride_dq_acc,
        .nhead_stride_dq = nhead_stride_dq,
        .nhead_stride_dk = nhead_stride_dk,
        .nhead_stride_dv = nhead_stride_dv,
        .nhead_stride_dbias = 0,
        .batch_stride_q = batch_stride_q,
        .batch_stride_k = batch_stride_k,
        .batch_stride_v = batch_stride_v,
        .batch_stride_bias = 0,
        .batch_stride_o = batch_stride_o,
        .batch_stride_randval = 0,
        .batch_stride_do = batch_stride_do,
        .batch_stride_lsed = batch_stride_lse,
        .batch_stride_dq_acc = batch_stride_dq_acc,
        .batch_stride_dq = batch_stride_dq,
        .batch_stride_dk = batch_stride_dk,
        .batch_stride_dv = batch_stride_dv,
        .batch_stride_dbias = 0,
        .split_stride_dq_acc = split_stride_dq_acc,
        .window_size_left = mask.left,
        .window_size_right = mask.right,
        .mask_type = static_cast<ck_tile::index_t>(mask.type),
        .p_drop = p_dropout,
        .p_undrop = p_undrop,
        .drop_seed_offset = std::move(drop_seed_var),
    };
#else
    return fmha_bwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         alibi_slopes_ptr,
                         out.data_ptr(),
                         softmax_lse.data_ptr(),
                         dout.data_ptr(),
                         d.data_ptr(),
                         nullptr,
                         dq.data_ptr(),
                         dk.data_ptr(),
                         dv.data_ptr(),
                         nullptr,
                         dq_acc.data_ptr(),
                         seqlens_q.data_ptr(),
                         seqlens_k.data_ptr(),
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         total_q,
                         total_k,
                         b,
                         max_seqlen_q,
                         max_seqlen_k,
                         hdim,
                         hdim,
                         h,
                         h_k,
                         softmax_scale,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_alibi_slopes,
                         stride_o,
                         0,
                         stride_do,
                         stride_dq_acc,
                         stride_dq,
                         stride_dk,
                         stride_dv,
                         0,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0,
                         nhead_stride_o,
                         0,
                         nhead_stride_do,
                         nhead_stride_lse,
                         nhead_stride_dq_acc,
                         nhead_stride_dq,
                         nhead_stride_dk,
                         nhead_stride_dv,
                         0,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0,
                         batch_stride_o,
                         0,
                         batch_stride_do,
                         batch_stride_lse,
                         batch_stride_dq_acc,
                         batch_stride_dq,
                         batch_stride_dk,
                         batch_stride_dv,
                         0,
                         split_stride_dq_acc,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         p_undrop,
                         drop_seed_offset};
#endif
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,                   // total_q x num_heads x head_size
               const at::Tensor &q,                      // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,                    // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,            // b x h x s   softmax logsumexp
               std::optional<at::Tensor> &dq_,           // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dk_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               std::optional<at::Tensor> &dv_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,           // b+1
               const at::Tensor &cu_seqlens_k,           // b+1
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k, // max sequence length to choose the kernel
               const float p_dropout,  // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float /*softcap*/,
               const bool deterministic,
               std::optional<at::Generator> gen_,
               std::optional<at::Tensor> &rng_state_)
{
#ifdef FLASHATTENTION_DISABLE_BACKWARD
    TORCH_CHECK(false, "This flash attention build does not support backward.");
#endif
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
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

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

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
    const int head_size = sizes[2];
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "CK FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    mask_info mask;
    if (is_causal) {
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // local
    }

    // q, k, v, out had been padded in mha_fwd
    // dq_, dk_, dv_ are also padded tensor
    CHECK_SHAPE(q, total_q, num_heads, head_size);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    CHECK_SHAPE(out, total_q, num_heads, head_size);
    CHECK_SHAPE(dout, total_q, num_heads, head_size);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

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

    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;

    if (!deterministic) {
        dq_accum = torch::zeros({num_heads, 1, total_q, head_size}, opts.dtype(at::kFloat));
    } else {
        const ck_tile::index_t kN0 = head_size <= 128 ? 128 : 64;
        const ck_tile::index_t nsplits = ck_tile::integer_divide_ceil(max_seqlen_k, kN0);
        dq_accum = torch::zeros({num_heads, nsplits, total_q, head_size}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if(zero_tensors) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    at::Tensor rng_state;

    if (rng_state_.has_value()) {
        rng_state = rng_state_.value();
    } else if(is_dropout) {
        rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0,
            philox_args, reinterpret_cast<uint64_t*>(rng_state.data_ptr()));
    } else {
        rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    }

    if (max_seqlen_q > 0) {
        std::pair<uint64_t*, uint64_t*> drop_seed_offset = {nullptr, nullptr};
        if (is_dropout) {
            auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
            drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        }
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_varlen_bwd_traits(mask, q_dtype_str, head_size, is_dropout, alibi_slopes_.has_value(), deterministic);

        auto args =
            get_ck_fmha_varlen_bwd_args(
                mask,
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                alibi_slopes_,
                out,
                softmax_lse,
                dout,
                dq_accum,
                softmax_d,
                dq,
                dk_expanded,
                dv_expanded,
                softmax_scale,
                p_dropout,
                drop_seed_offset);

        float t = fmha_bwd(traits, args, stream_config);
        TORCH_CHECK(t >= 0, "invalid argument for fmha_bwd");
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size}), {2});
    }

    return { dq, dk, dv, softmax_d };
}
