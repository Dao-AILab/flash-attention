/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "ck_build_config.hpp"
#include "fmha_bwd.hpp"
#include "mask.hpp"
#include <variant>

fmha_bwd_traits get_ck_fmha_bwd_traits(const mask_info &mask,
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
        .is_group_mode = false,
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
                           false,
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           false,
                           has_dropout,
                           false,
                           deterministic};
#endif
}

fmha_bwd_args get_ck_fmha_bwd_args(const mask_info &mask,
                                   // sizes
                                   const int b,
                                   const int seqlen_q,
                                   const int seqlen_k,
                                   const int h,
                                   const int h_k,
                                   const int hdim,
                                   // device pointers
                                   const at::Tensor q,
                                   const at::Tensor k,
                                   const at::Tensor v,
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
    // q: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_q = q.stride(0);
    ck_tile::index_t stride_q = q.stride(1);
    ck_tile::index_t nhead_stride_q = q.stride(2);

    // k: (batch_size, seqlen_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_k = k.stride(0);
    ck_tile::index_t stride_k = k.stride(1);
    ck_tile::index_t nhead_stride_k = k.stride(2);

    // v: (batch_size, seqlen_k, nheads_k, hdim)
    ck_tile::index_t batch_stride_v = v.stride(0);
    ck_tile::index_t stride_v = v.stride(1);
    ck_tile::index_t nhead_stride_v = v.stride(2);

    // o: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_o = out.stride(0);
    ck_tile::index_t stride_o = out.stride(1);
    ck_tile::index_t nhead_stride_o = out.stride(2);

    // lse: (batch_size, nheads, seqlen_q)
    ck_tile::index_t batch_stride_lse = softmax_lse.stride(0);
    ck_tile::index_t nhead_stride_lse = softmax_lse.stride(1);

    // do: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_do = dout.stride(0);
    ck_tile::index_t stride_do = dout.stride(1);
    ck_tile::index_t nhead_stride_do = dout.stride(2);

    // d: (batch_size, nheads, seqlen_q)
    // CK assume d share the same stride with lse

    // dq: (batch_size, seqlen_q, nheads, hdim)
    ck_tile::index_t batch_stride_dq = dq.stride(0);
    ck_tile::index_t stride_dq = dq.stride(1);
    ck_tile::index_t nhead_stride_dq = dq.stride(2);

    // dk_expanded: (batch_size, seqlen_k, nheads, hdim)
    ck_tile::index_t batch_stride_dk = dk.stride(0);
    ck_tile::index_t stride_dk = dk.stride(1);
    ck_tile::index_t nhead_stride_dk = dk.stride(2);

    // dv_expanded: (batch_size, seqlen_k, nheads, hdim)
    ck_tile::index_t batch_stride_dv = dv.stride(0);
    ck_tile::index_t stride_dv = dv.stride(1);
    ck_tile::index_t nhead_stride_dv = dv.stride(2);

    // dq_acc: (batch_size, nheads, split, seqlen_q, hdim)
    ck_tile::index_t batch_stride_dq_acc = dq_acc.stride(0);
    ck_tile::index_t nhead_stride_dq_acc = dq_acc.stride(1);
    ck_tile::index_t split_stride_dq_acc = dq_acc.stride(2);
    ck_tile::index_t stride_dq_acc = dq_acc.stride(3);

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
        .seqstart_q_ptr = nullptr,
        .seqstart_k_ptr = nullptr,
        .seqlen_q_ptr = nullptr,
        .seqlen_k_ptr = nullptr,
        .cu_seqlen_q_ptr = nullptr,
        .cu_seqlen_k_ptr = nullptr,
        .seqlen_q = seqlen_q,
        .seqlen_k = seqlen_k,
        .batch = b,
        .max_seqlen_q = seqlen_q,
        .max_seqlen_k = seqlen_k,
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
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         nullptr,
                         seqlen_q,
                         seqlen_k,
                         b,
                         seqlen_q,
                         seqlen_k,
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
mha_bwd(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x multiple_of(head_size, 8)
        const at::Tensor &q,                      // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,                    // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,            // b x h x seqlen_q
        std::optional<at::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        std::optional<at::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,                    // probability to drop
        const float softmax_scale,
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
#ifdef HIPIFY_V2
    auto stream = at::cuda::getCurrentCUDAStream().stream();
#else
    auto stream = at::cuda::getCurrentHIPStream().stream();
#endif

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

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
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "CK FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= seqlen_k) { window_size_left = -1; }
    if (window_size_right >= seqlen_k) { window_size_right = -1; }

    mask_info mask;
    if (is_causal) {
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", seqlen_q, seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k); // local
    }

    // q, k, v, out had been padded in mha_fwd
    // dq_, dk_, dv_ are also padded tensor
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);

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

    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;

    if (!deterministic) {
        dq_accum = torch::zeros({batch_size, num_heads, 1, seqlen_q, head_size}, opts.dtype(at::kFloat));
    } else {
        const ck_tile::index_t kN0 = head_size <= 128 ? 128 : 64;
        const ck_tile::index_t nsplits = ck_tile::integer_divide_ceil(seqlen_k, kN0);
        dq_accum = torch::zeros({batch_size, num_heads, nsplits, seqlen_q, head_size}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
        dv_expanded = torch::empty({batch_size, seqlen_k, num_heads, head_size}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
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
    }

    if (seqlen_q > 0) {
        std::pair<uint64_t*, uint64_t*> drop_seed_offset = {nullptr, nullptr};
        if (is_dropout) {
            auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
            drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        }
        ck_tile::stream_config stream_config{stream};

        auto traits =
            get_ck_fmha_bwd_traits(mask, q_dtype_str, head_size, is_dropout, alibi_slopes_.has_value(), deterministic);

        auto args =
            get_ck_fmha_bwd_args(
                mask,
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                q,
                k,
                v,
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
        at::sum_out(dk, at::reshape(dk_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
        at::sum_out(dv, at::reshape(dv_expanded, {batch_size, seqlen_k, num_heads_k, num_heads / num_heads_k, head_size}), {3});
    }

    return { dq, dk, dv, softmax_d };
}
