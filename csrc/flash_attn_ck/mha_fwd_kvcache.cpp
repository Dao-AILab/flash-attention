/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

#include "fmha_fwd.hpp"
#include "rotary.hpp"

fmha_fwd_appendkv_traits get_ck_fmha_fwd_appendkv_traits(std::string dtype,
                                                        int head_size,
                                                        int rotary_dim,
                                                        bool is_rotary_interleaved)
{
    rope_enum rope_type = (0 < rotary_dim ? (is_rotary_interleaved ? rope_enum::interleaved
                                                                   : rope_enum::half_rotated)
                                          : rope_enum::none);

    return fmha_fwd_appendkv_traits{head_size,
                                    head_size,
                                    dtype,
                                    true,  // is_v_rowmajor
                                    rope_type};
}

fmha_fwd_splitkv_traits get_ck_fmha_fwd_splitkv_traits(const mask_info &mask,
                                                       std::string dtype,
                                                       int head_size,
                                                       bool has_lse,
                                                       bool enable_alibi)
{
    return fmha_fwd_splitkv_traits{head_size,
                                   head_size,
                                   dtype,
                                   false, // is_group_mode
                                   true, // is_v_rowmajor
                                   mask.type,
                                   enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
                                   has_lse,
                                   false}; // do_fp8_static_quant
}

fmha_fwd_appendkv_args get_ck_fmha_fwd_appendkv_args(const int b,
                                                     const int seqlen_q,
                                                     const int seqlen_knew,
                                                     const int h,
                                                     const int h_k,
                                                     const int d,
                                                     const int rotary_dim,
                                                     const bool has_mask,
                                                     const int page_block_size,
                                                     // device pointers
                                                     const at::Tensor q,
                                                     const at::Tensor kcache,
                                                     const at::Tensor vcache,
                                                     const at::Tensor knew,
                                                     const at::Tensor vnew,
                                                     std::optional<const at::Tensor> &seqlens_k_,
                                                     std::optional<const at::Tensor> &rotary_cos_,
                                                     std::optional<const at::Tensor> &rotary_sin_,
                                                     std::optional<const at::Tensor> &cache_batch_idx_,
                                                     std::optional<at::Tensor> &block_table_)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // kcache: (batch_size_c, seqlen_k, nheads_k, d) or (num_blocks, page_block_size, nheads_k, d)
    // vcache: (batch_size_c, seqlen_k, nheads_k, d) or (num_blocks, page_block_size, nheads_k, d)
    // knew: (batch_size, seqlen_knew, nheads_k, d)
    // vnew: (batch_size, seqlen_knew, nheads_k, d)

    // seqlens_k: (batch_size)
    // rotary_cos: (seqlen_ro, rotary_dim / 2)
    // rotary_sin: (seqlen_ro, rotary_dim / 2)
    // block_table: (batch_size, max_num_blocks_per_seq)

    fmha_fwd_appendkv_args args;
    args.q_ptr = q.data_ptr();
    args.k_ptr = kcache.data_ptr();
    args.knew_ptr = knew.data_ptr();
    args.v_ptr = vcache.data_ptr();
    args.vnew_ptr = vnew.data_ptr();
    args.seqlen_k_ptr = seqlens_k_.has_value() ? seqlens_k_.value().data_ptr() : nullptr;

    args.seqlen_q = seqlen_q;
    args.seqlen_knew = seqlen_knew;
    args.batch = b;
    args.hdim_q = d;
    args.hdim_v = d;
    args.nhead_q = h;
    args.nhead_k = h_k;

    args.rotary_cos_ptr = rotary_cos_.has_value() ? rotary_cos_.value().data_ptr() : nullptr;
    args.rotary_sin_ptr = rotary_sin_.has_value() ? rotary_sin_.value().data_ptr() : nullptr;
    args.rotary_dim = rotary_dim;
    args.has_mask = has_mask;

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

    args.cache_batch_idx = cache_batch_idx_.has_value() ?
        reinterpret_cast<int *>(cache_batch_idx_.value().data_ptr()) : nullptr;

    args.batch_stride_q = q.stride(0);
    args.stride_q = q.stride(1);
    args.nhead_stride_q = q.stride(2);

    args.batch_stride_k = kcache.stride(0);
    args.stride_k = kcache.stride(1);
    args.nhead_stride_k = kcache.stride(2);

    args.batch_stride_knew = knew.stride(0);
    args.stride_knew = knew.stride(1);
    args.nhead_stride_knew = knew.stride(2);

    args.batch_stride_v = vcache.stride(0);
    args.stride_v = vcache.stride(1);
    args.nhead_stride_v = vcache.stride(2);

    args.batch_stride_vnew = vnew.stride(0);
    args.stride_vnew = vnew.stride(1);
    args.nhead_stride_vnew = vnew.stride(2);

    return args;
}

fmha_fwd_splitkv_args get_ck_fmha_fwd_splitkv_args(bool has_lse,
                                                   const mask_info &mask,
                                                   const int b,
                                                   const int seqlen_q,
                                                   const int seqlen_k,
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
                                                   const at::Tensor seqlens_k,
                                                   std::optional<const at::Tensor> &cache_batch_idx_,
                                                   std::optional<at::Tensor> &block_table_,
                                                   std::optional<at::Tensor> &alibi_slopes_,
                                                   at::Tensor out,
                                                   at::Tensor lse,
                                                   at::Tensor lse_acc,
                                                   at::Tensor out_acc)
{
    // q: (batch_size, seqlen_q, nheads, d)
    // k: (batch_size, seqlen_k, nheads_k, d)
    // v: (batch_size, seqlen_k, nheads_k, d)
    // o: (batch_size, seqlen_q, nheads, d)

    // alibi_slopes:(batch_size, nheads) or (nhead)
    // lse: (batch_size, nheads, seqlen_q)
    // lse_acc: (split, batch_size, nheads, seqlen_q)
    // o_acc: (split, batch_size, nheads, seqlen_q, d)

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

    args.cache_batch_idx = cache_batch_idx_.has_value() ? cache_batch_idx_.value().data_ptr() : nullptr;

    args.seqstart_q_ptr = nullptr;
    args.seqstart_k_ptr = nullptr;
    args.seqlen_k_ptr = seqlens_k.data_ptr();

    args.seqlen_q = seqlen_q;
    args.seqlen_k = seqlen_k;
    args.batch = b;
    args.max_seqlen_q = seqlen_q;
    args.hdim_q = d;
    args.hdim_v = d;
    args.nhead_q = h;
    args.nhead_k = h_k;
    args.num_splits = num_splits;

    args.scale_s = softmax_scale;
    args.scale_p = 1;
    args.scale_o = 1;

    args.batch_stride_q = q.stride(0);
    args.stride_q = q.stride(1);
    args.nhead_stride_q = q.stride(2);

    args.batch_stride_k = k.stride(0);
    args.stride_k = k.stride(1);
    args.nhead_stride_k = k.stride(2);

    args.batch_stride_v = v.stride(0);
    args.stride_v = v.stride(1);
    args.nhead_stride_v = v.stride(2);

    args.batch_stride_o = out.stride(0);
    args.stride_o = out.stride(1);
    args.nhead_stride_o = out.stride(2);

    args.batch_stride_bias = 0;
    args.stride_bias = 0;
    args.nhead_stride_bias = 0;

    args.batch_stride_lse = 0;
    args.nhead_stride_lse = 0;

    args.split_stride_lse_acc = lse_acc.stride(0);
    args.batch_stride_lse_acc = lse_acc.stride(1);
    args.nhead_stride_lse_acc = lse_acc.stride(2);

    args.split_stride_o_acc = out_acc.stride(0);
    args.batch_stride_o_acc = out_acc.stride(1);
    args.nhead_stride_o_acc = out_acc.stride(2);
    args.stride_o_acc = out_acc.stride(3);

    if (has_lse) {
        args.lse_ptr = lse.data_ptr();
        args.batch_stride_lse = lse.stride(0);
        args.nhead_stride_lse = lse.stride(1);
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
mha_fwd_kvcache(at::Tensor &q,                                      // batch_size x seqlen_q x num_heads x head_size
                const at::Tensor &kcache,                           // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                const at::Tensor &vcache,                           // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
                std::optional<const at::Tensor> &k_,                // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &v_,                // batch_size x seqlen_knew x num_heads_k x head_size
                std::optional<const at::Tensor> &seqlens_k_,        // batch_size
                std::optional<const at::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
                std::optional<const at::Tensor> &cache_batch_idx_, // indices to index into the KV cache
                std::optional<const at::Tensor> & /*leftpad_k_*/,   // batch_size
                std::optional<at::Tensor> &block_table_,            // batch_size x max_num_blocks_per_seq
                std::optional<at::Tensor> &alibi_slopes_,           // num_heads or batch_size x num_heads
                std::optional<at::Tensor> &out_,                    // batch_size x seqlen_q x num_heads x head_size
                const float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                const float /*softcap*/,
                bool is_rotary_interleaved, // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
                int num_splits)
{
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value must have the same dtype");
    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

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
    TORCH_CHECK(!paged_KV || page_block_size % 128 == 0, "Paged KV cache block size must be divisible by 128");
    const int seqlen_k = !paged_KV ? kcache.size(1) : max_num_blocks_per_seq * page_block_size;
    const int num_heads_k = kcache.size(2);
    const int batch_size_c = !paged_KV ? kcache.size(0) : batch_size;
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_og <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // causal=true is the same as causal=false in this case
    if (seqlen_q == 1 && !alibi_slopes_.has_value()) { is_causal = false; }
    if (is_causal) { window_size_right = 0; }

    mask_info mask;
    if (is_causal) {
        // Causal is the special case where window_size_right == 0 and window_size_left < 0.
        window_size_right = 0;
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

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in this case
    // H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k && window_size_left < 0 && window_size_right < 0 && head_size_og % 8 == 0 && !alibi_slopes_.has_value();
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
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0) { out = torch::empty_like(q_padded); }
    } else {
        out = torch::empty_like(q_padded);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_8x = round_multiple(head_size_og, 8);

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();

    // TODO - check gradient, only training require lse
    bool has_lse = true;
    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    int seqlen_knew = 0;
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
        seqlen_knew = k.size(1);
        CHECK_SHAPE(k, batch_size, seqlen_knew, num_heads_k, head_size_og);
        CHECK_SHAPE(v, batch_size, seqlen_knew, num_heads_k, head_size_og);
        if (head_size_og % 8 != 0) {
            k_padded = torch::nn::functional::pad(k, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            v_padded = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        } else {
            k_padded = k;
            v_padded = v;
        }
    }

    if (seqlens_k_.has_value()) {
        auto seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
        CHECK_DEVICE(seqlens_k);
        CHECK_CONTIGUOUS(seqlens_k);
        CHECK_SHAPE(seqlens_k, batch_size);
    }

    int rotary_dim = 0;
    if (rotary_cos_.has_value()) {
        TORCH_CHECK(k_.has_value(), "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided");
        auto rotary_cos = rotary_cos_.value();
        CHECK_DEVICE(rotary_cos);
        rotary_dim = rotary_cos.size(1) * 2;
        TORCH_CHECK(rotary_dim <= head_size_og, "rotary_dim must be <= headdim");
        TORCH_CHECK(rotary_dim % 16 == 0, "Only rotary dimensions divisible by 16 are currently supported");
        const int seqlen_ro = rotary_cos.size(0);
        TORCH_CHECK(seqlen_ro >= seqlen_k, "cos/sin seqlen must be at least the seqlen of KV cache");
        CHECK_SHAPE(rotary_cos, seqlen_ro, rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_cos);
        TORCH_CHECK(rotary_cos.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");

        TORCH_CHECK(rotary_sin_.has_value(), "If rotary cos is provided, rotary sin must also be provided");
        auto rotary_sin = rotary_sin_.value();
        CHECK_DEVICE(rotary_sin);
        CHECK_SHAPE(rotary_sin, seqlen_ro, rotary_dim / 2);
        CHECK_CONTIGUOUS(rotary_sin);
        TORCH_CHECK(rotary_sin.scalar_type() == q_dtype, "rotary_cos must have the same dtype as query");
    }


    if (cache_batch_idx_.has_value()) {
        auto cache_batch_idx = cache_batch_idx_.value();
        CHECK_DEVICE(cache_batch_idx);
        CHECK_CONTIGUOUS(cache_batch_idx);
        TORCH_CHECK(cache_batch_idx.scalar_type() == torch::kInt32, "cache_batch_idx must have dtype int32");
    }

    num_splits = flash::override_num_splits_if_necessary(batch_size, num_heads, seqlen_q, head_size_8x, 0, num_splits);
    TORCH_CHECK(num_splits > 0, "num_splits should greater than 0");
    TORCH_CHECK(num_splits <= 128, "num_splits greater than 128 is not supported");

    // Keep references to these tensors to extend their lifetime
    auto softmax_lse_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    auto out_accum = torch::empty({num_splits, batch_size, num_heads, seqlen_q, head_size_8x}, opts.dtype(at::kFloat));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    ck_tile::stream_config stream_config{stream};

    if (seqlen_knew > 0 || rotary_dim > 0) {
        auto appendkv_traits =
            get_ck_fmha_fwd_appendkv_traits(q_dtype_str, head_size_8x, rotary_dim, is_rotary_interleaved);

        auto appendkv_args =
            get_ck_fmha_fwd_appendkv_args(
                batch_size,
                seqlen_q,
                seqlen_knew,
                num_heads,
                num_heads_k,
                head_size_8x,
                rotary_dim,
                mask.type != mask_enum::no_mask,
                page_block_size,
                q_padded,
                kcache_padded,
                vcache_padded,
                k_padded,
                v_padded,
                seqlens_k_,
                rotary_cos_,
                rotary_sin_,
                cache_batch_idx_,
                block_table_);

        fmha_fwd_appendkv(appendkv_traits, appendkv_args, stream_config);
    }

    // seqlens_k_ is the seqlen of kvcache. We need to add seqlen_knew for before attention
    auto append_seqlens_k = torch::empty({batch_size}, opts.dtype(torch::kInt32));
    if (seqlens_k_.has_value())
        append_seqlens_k = seqlens_k_.value() + seqlen_knew;
    else
        append_seqlens_k.fill_(seqlen_knew);

    // we use splitkv even num_splits == 1, because fmha_fwd() does not support seqlen_k_ in batch mode
    auto splitkv_traits =
        get_ck_fmha_fwd_splitkv_traits(mask, q_dtype_str, head_size_8x, has_lse, alibi_slopes_.has_value());

    auto splitkv_args =
        get_ck_fmha_fwd_splitkv_args(
            has_lse,
            mask,
            batch_size,
            seqlen_q,
            seqlen_k,
            num_heads,
            num_heads_k,
            head_size_8x,
            page_block_size,
            num_splits,
            softmax_scale,
            q_padded,
            kcache_padded,
            vcache_padded,
            append_seqlens_k,
            cache_batch_idx_,
            block_table_,
            alibi_slopes_,
            out,
            softmax_lse,
            softmax_lse_accum,
            out_accum);

    fmha_fwd_splitkv(splitkv_traits, splitkv_args, stream_config);

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
