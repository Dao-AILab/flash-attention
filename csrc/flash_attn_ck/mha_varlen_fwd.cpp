/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "fmha_fwd.hpp"
#include "mask.hpp"

fmha_fwd_traits get_ck_fmha_varlen_fwd_traits(const mask_info &mask,
                                              std::string dtype,
                                              int head_size,
                                              bool has_dropout,
                                              bool has_lse,
                                              bool enable_alibi)
{
    return fmha_fwd_traits{head_size,
                           head_size,
                           dtype,
                           true, // is_group_mode
                           true, // is_v_rowmajor
                           mask.type,
                           enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                           has_lse,
                           has_dropout,
                           false}; // do_fp8_static_quant
}

fmha_fwd_splitkv_traits get_ck_fmha_varlen_fwd_splitkv_traits(const mask_info &mask,
                                                              std::string dtype,
                                                              int head_size,
                                                              bool has_lse,
                                                              bool enable_alibi)
{
    return fmha_fwd_splitkv_traits{head_size,
                                   head_size,
                                   dtype,
                                   true, // is_group_mode
                                   true, // is_v_rowmajor
                                   mask.type,
                                   enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                                   has_lse,
                                   false}; // do_fp8_static_quant
}

fmha_fwd_args get_ck_fmha_varlen_fwd_args(bool has_lse,
                                          bool has_dropout_randval,
                                          const mask_info &mask,
                                          // sizes
                                          const int b,
                                          const int max_seqlen_q,
                                          const int h,
                                          const int h_k,
                                          const int d,
                                          // device pointers
                                          const at::Tensor q,
                                          const at::Tensor k,
                                          const at::Tensor v,
                                          const at::Tensor seqlens_q,
                                          const at::Tensor seqlens_k,
                                          std::optional<at::Tensor> &alibi_slopes_,
                                          at::Tensor out,
                                          at::Tensor softmax_lse,
                                          at::Tensor dropout_randval,
                                          float softmax_scale,
                                          float p_dropout,
                                          std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    // q: (total_q, nheads, d)
    // k: (total_k, nheads_k, d)
    // v: (total_k, nheads_k, d)
    // o: (total_q, nheads, d)

    // alibi_slopes:(batch, nheads) or (nhead)
    // lse: (nheads, total_q)
    // randval: (nheads, total_q, max_seqlen_k)

    ck_tile::index_t total_q = q.size(0);
    ck_tile::index_t total_k = k.size(0);

    ck_tile::index_t stride_q = q.stride(0);
    ck_tile::index_t stride_k = k.stride(0);
    ck_tile::index_t stride_v = v.stride(0);
    ck_tile::index_t stride_o = out.stride(0);
    ck_tile::index_t stride_randval = has_dropout_randval ? dropout_randval.stride(1) : 0;

    ck_tile::index_t nhead_stride_q = q.stride(1);
    ck_tile::index_t nhead_stride_k = k.stride(1);
    ck_tile::index_t nhead_stride_v = v.stride(1);
    ck_tile::index_t nhead_stride_o = out.stride(1);
    ck_tile::index_t nhead_stride_lse = has_lse ? softmax_lse.stride(0) : 0;
    ck_tile::index_t nhead_stride_randval = has_dropout_randval ? dropout_randval.stride(0) : 0;

    ck_tile::index_t batch_stride_q = 0;
    ck_tile::index_t batch_stride_k = 0;
    ck_tile::index_t batch_stride_v = 0;
    ck_tile::index_t batch_stride_o = 0;
    ck_tile::index_t batch_stride_lse = 0;
    ck_tile::index_t batch_stride_randval = 0;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_fwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         alibi_slopes_ptr, // bias
                         has_dropout_randval ? dropout_randval.data_ptr() : nullptr,
                         has_lse ? softmax_lse.data_ptr() : nullptr,
                         out.data_ptr(),
                         seqlens_q.data_ptr(), // seqstart_q
                         seqlens_k.data_ptr(), // seqstart_k
                         nullptr,              // seqlen_kpads
                         total_q,
                         total_k,
                         b,
                         max_seqlen_q,
                         d,             // hdim_q
                         d,             // hdim_v
                         h,             // nhead
                         h_k,           // nhead_k
                         softmax_scale, // scale_s
                         1,             // scale_p
                         1,             // scale_o
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_alibi_slopes,
                         stride_randval,
                         stride_o,
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias, FA without bias
                         nhead_stride_randval,
                         nhead_stride_lse,
                         nhead_stride_o,
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0, // batch_stride_bias, FA without bias
                         batch_stride_randval,
                         batch_stride_lse,
                         batch_stride_o,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         has_dropout_randval,
                         drop_seed_offset};
}

fmha_fwd_splitkv_args get_ck_fmha_varlen_fwd_splitkv_args(bool has_lse,
                                                          const mask_info &mask,
                                                          const int b,
                                                          const int max_seqlen_q,
                                                          const int h,
                                                          const int h_k,
                                                          const int d,
                                                          const int page_block_size,
                                                          const int num_splits,
                                                          float softmax_scale,
                                                          // device pointers
                                                          const at::Tensor q,
                                                          const at::Tensor k,
                                                          const at::Tensor v,
                                                          const at::Tensor seqlens_q,
                                                          const at::Tensor seqlens_k,
                                                          std::optional<at::Tensor> &block_table_,
                                                          std::optional<at::Tensor> &alibi_slopes_,
                                                          at::Tensor out,
                                                          at::Tensor lse,
                                                          at::Tensor lse_acc,
                                                          at::Tensor out_acc)
{
    // q: (total_q, nheads, d)
    // k: (num_blocks, page_block_size, num_heads_k, d)
    // v: (num_blocks, page_block_size, num_heads_k, d)
    // o: (total_q, nheads, d)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (nheads, total_q)
    // lse_acc: (nheads, split, total_q)
    // o_acc: (nheads, split, total_q, d)
    // block_table: (batch_size, max_num_blocks_per_seq)

    fmha_fwd_splitkv_args args;
    args.q_ptr = q.data_ptr();
    args.k_ptr = k.data_ptr();
    args.v_ptr = v.data_ptr();
    args.bias_ptr = nullptr;
    args.lse_acc_ptr = lse_acc.data_ptr();
    args.o_acc_ptr = out_acc.data_ptr();
    args.lse_ptr = nullptr;
    args.o_ptr = out.data_ptr();

    if (block_table_.has_value())
    {
        auto block_table = block_table_.value();
        args.block_table_ptr = block_table.data_ptr();
        args.batch_stride_block_table = block_table.stride(0);
        args.page_block_size = page_block_size;
    }
    else
    {
        args.block_table_ptr = nullptr;
        args.batch_stride_block_table = 0;
        args.page_block_size = 0;
    }

    args.is_gappy = false;
    args.cache_batch_idx = nullptr;

    args.seqstart_q_ptr = seqlens_q.data_ptr();
    args.seqstart_k_ptr = seqlens_k.data_ptr();
    args.seqlen_k_ptr = nullptr;

    args.batch = b;
    args.max_seqlen_q = max_seqlen_q;
    args.hdim_q = d;
    args.hdim_v = d;
    args.nhead_q = h;
    args.nhead_k = h_k;
    args.num_splits = num_splits;

    args.scale_s = softmax_scale;
    args.scale_p = 1;
    args.scale_o = 1;

    args.batch_stride_q = 0;
    args.stride_q = q.stride(0);
    args.nhead_stride_q = q.stride(1);

    args.batch_stride_k = k.stride(0);
    args.stride_k = k.stride(1);
    args.nhead_stride_k = k.stride(2);

    args.batch_stride_v = v.stride(0);
    args.stride_v = v.stride(1);
    args.nhead_stride_v = v.stride(2);

    args.batch_stride_o = 0;
    args.stride_o = out.stride(0);
    args.nhead_stride_o = out.stride(1);

    args.batch_stride_bias = 0;
    args.stride_bias = 0;
    args.nhead_stride_bias = 0;

    args.batch_stride_lse = 0;
    args.nhead_stride_lse = 0;

    args.batch_stride_lse_acc = 0;
    args.nhead_stride_lse_acc = lse_acc.stride(0);
    args.split_stride_lse_acc = lse_acc.stride(1);

    args.batch_stride_o_acc = 0;
    args.nhead_stride_o_acc = out_acc.stride(0);
    args.split_stride_o_acc = out_acc.stride(1);
    args.stride_o_acc = out_acc.stride(2);

    if (has_lse) {
        args.lse_ptr = lse.data_ptr();
        args.batch_stride_lse = 0;
        args.nhead_stride_lse = lse.stride(0);
    }

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        args.bias_ptr = alibi_slopes.data_ptr();
        args.stride_bias = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    args.window_size_left = mask.left;
    args.window_size_right = mask.right;
    args.mask_type = static_cast<ck_tile::index_t>(mask.type);

    return args;
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,  // b+1
               const at::Tensor &cu_seqlens_k,  // b+1
               std::optional<at::Tensor> & /*seqused_k*/,
               std::optional<const at::Tensor> &/*leftpad_k_*/, // batch_size
               std::optional<at::Tensor> &block_table_,  // batch_size x max_num_blocks_per_seq
               std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float /*softcap*/,
               const bool return_dropout_randval,
               std::optional<at::Generator> gen_)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(cu_seqlens_q);
    CHECK_DEVICE(cu_seqlens_k);

    at::Tensor block_table;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        CHECK_DEVICE(block_table);
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype torch.int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = paged_KV ? k.size(2) : k.size(1);

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 1 : k.size(1);
    TORCH_CHECK(!paged_KV || page_block_size % 128 == 0, "Paged KV cache block size must be divisible by 128");

    if (max_seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }  // causal=true is the same as causal=false in this case

    // TODO
    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza

    const int total_q = q.size(0);

    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size <= 256, "CK only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    mask_info mask;

    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
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

    CHECK_SHAPE(q, total_q, num_heads, head_size);
    if (!paged_KV) {
        const int total_k = k.size(0);
        CHECK_SHAPE(k, total_k, num_heads_k, head_size);
        CHECK_SHAPE(v, total_k, num_heads_k, head_size);
    } else {
        CHECK_SHAPE(k, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(v, num_blocks, page_block_size, num_heads_k, head_size);
        CHECK_SHAPE(block_table, batch_size, max_num_blocks_per_seq);
    }

    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total_q, num_heads, head_size);
    }
    else {
        out = torch::empty_like(q);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    bool has_lse = true;
    bool has_dropout = p_dropout > 0.0f;
    if (has_dropout)
        TORCH_CHECK(!paged_KV, "Paged KV does not support dropout");

    at::Tensor softmax_lse;
    // TODO - check gradient, only training require lse
    softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(torch::kFloat32));

    at::Tensor p;
    if (return_dropout_randval) {
        TORCH_CHECK(has_dropout, "return_dropout_randval require p_dropout > 0");
        p = torch::empty({num_heads, total_q, max_seqlen_k}, opts.dtype(torch::kUInt8));
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    if (zero_tensors)
    {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_dropout_randval) {p.zero_();}
    }

    int num_splits = 0;
    num_splits = flash::override_num_splits_if_necessary(batch_size, num_heads, max_seqlen_q, head_size, 0, num_splits);
    TORCH_CHECK(num_splits > 0, "num_splits should greater than 0");
    TORCH_CHECK(num_splits <= 128, "num_splits greater than 128 is not supported");

    auto softmax_lse_accum = torch::empty({num_heads, num_splits, total_q}, opts.dtype(at::kFloat));
    auto out_accum = torch::empty({num_heads, num_splits, total_q, head_size}, opts.dtype(at::kFloat));

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    auto rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    if (p_dropout > 0.0)  {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_, at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            flash::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0, philox_args, rng_state_ptr);
    }

    if (max_seqlen_k > 0) {
        auto stream = at::cuda::getCurrentHIPStream().stream();
        ck_tile::stream_config stream_config{stream};

        if (paged_KV)
        {
            auto traits =
                get_ck_fmha_varlen_fwd_splitkv_traits(
                    mask,
                    q_dtype_str,
                    head_size,
                    has_lse,
                    alibi_slopes_.has_value());

            auto args =
                get_ck_fmha_varlen_fwd_splitkv_args(
                    has_lse,
                    mask,
                    batch_size,
                    max_seqlen_q,
                    num_heads,
                    num_heads_k,
                    head_size,
                    page_block_size,
                    num_splits,
                    softmax_scale,
                    q,
                    k,
                    v,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    block_table_,
                    alibi_slopes_,
                    out,
                    softmax_lse,
                    softmax_lse_accum,
                    out_accum);

            float t = fmha_fwd_splitkv(traits, args, stream_config);
            TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd_splitkv");
        }
        else
        {
            auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);

            auto traits =
                get_ck_fmha_varlen_fwd_traits(
                    mask,
                    q_dtype_str,
                    head_size,
                    has_dropout,
                    has_lse,
                    alibi_slopes_.has_value());

            auto args =
                get_ck_fmha_varlen_fwd_args(
                    has_lse,
                    return_dropout_randval,
                    mask,
                    batch_size,
                    max_seqlen_q,
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
                    p,
                    softmax_scale,
                    p_dropout,
                    drop_seed_offset);

            float t = fmha_fwd(traits, args, stream_config);
            TORCH_CHECK(t >= 0, "invalid argument for fmha_fwd");
        }
    }
    else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}
