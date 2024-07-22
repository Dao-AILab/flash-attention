/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#include "flash_common.hpp"

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,
        const at::Tensor &k,
        const at::Tensor &v,
        c10::optional<at::Tensor> &out_,
        c10::optional<at::Tensor> &alibi_slopes_,
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        c10::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,                            // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               const at::Tensor &v,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
               c10::optional<at::Tensor> &out_,          // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,           // b+1
               const at::Tensor &cu_seqlens_k,           // b+1
               c10::optional<at::Tensor> &seqused_k,     // b. If given, only this many elements of each batch element's keys are used.
               c10::optional<const at::Tensor> &leftpad_k_, // batch_size
               c10::optional<at::Tensor> &block_table_,  // batch_size x max_num_blocks_per_seq
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool return_softmax,
               c10::optional<at::Generator> gen_);

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,                   // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,                      // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,                      // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,                    // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_lse,            // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,           // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,           // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,           // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,                    // probability to drop
        const float softmax_scale,
        const bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool deterministic,
        c10::optional<at::Generator> gen_,
        c10::optional<at::Tensor> &rng_state);

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,                   // total_q x num_heads x head_size
               const at::Tensor &q,                      // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               const at::Tensor &k,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &v,                      // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &out,                    // total_q x num_heads x head_size
               const at::Tensor &softmax_lse,            // b x h x s   softmax logsumexp
               c10::optional<at::Tensor> &dq_,           // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dk_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               c10::optional<at::Tensor> &dv_,           // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
               const at::Tensor &cu_seqlens_q,           // b+1
               const at::Tensor &cu_seqlens_k,           // b+1
               c10::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
               const int max_seqlen_q,
               const int max_seqlen_k, // max sequence length to choose the kernel
               const float p_dropout,  // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const float softcap,
               const bool deterministic,
               c10::optional<at::Generator> gen_,
               c10::optional<at::Tensor> &rng_state);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable length)");
}
