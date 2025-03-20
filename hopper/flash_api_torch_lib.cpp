#include "registration.h"
#include "pytorch_shim.h"

#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

/**
 *  Externs for the flash_attn ops to be exposed as a pytorch library
 */

// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,   // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
        const at::Tensor &k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d) if there is page_table.
        const at::Tensor &v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k, dv) if there is page_table.
        std::optional<const at::Tensor> &k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
        std::optional<const at::Tensor> &v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
        std::optional<const at::Tensor> &q_v_,  // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
        std::optional<at::Tensor> &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
        std::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const at::Tensor> &seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
        std::optional<int> max_seqlen_q_,
        // TODO: check if we need max_seqlen_k
        std::optional<int> max_seqlen_k_,
        std::optional<const at::Tensor> &page_table_, // (b_k, max_num_pages_per_seq)
        std::optional<const at::Tensor> &kv_batch_idx_, // b. indices to index into the KV cache
        std::optional<const at::Tensor> &leftpad_k_, // b
        std::optional<const at::Tensor> &rotary_cos_, // seqlen_ro x (rotary_dim / 2)
        std::optional<const at::Tensor> &rotary_sin_, // seqlen_ro x (rotary_dim / 2)
        std::optional<at::Tensor> &q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> &k_descale_,  // (b, h_k)
        std::optional<at::Tensor> &v_descale_,  // (b, h_k)
        float const softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        float const softcap,
        bool const is_rotary_interleaved,   // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
        std::optional<at::Tensor> &scheduler_metadata_,  // (b + 1)
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin
);

// Only applicable to the case where seqused_k (i.e. cache_seqlens) is available
at::Tensor
mha_fwd_get_scheduler_metadata(
        int batch_size,
        int max_seqlen_q,
        int max_seqlen_k,
        int num_heads,
        int num_heads_k,
        int headdim,
        int headdim_v,
        at::ScalarType qkv_dtype,
        const at::Tensor &seqused_k, // b
        std::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_,  // b+1
        std::optional<const at::Tensor> &cu_seqlens_k_new_,  // b+1
        std::optional<const at::Tensor> &seqused_q_, // b. If given, only this many elements of each batch element's queries and outputs are used.
        std::optional<const at::Tensor> &leftpad_k_, // b
        std::optional<int> page_size,
        int max_seqlen_k_new,  // 0 means we're not appending new KV
        bool is_causal,
        int window_size_left,
        int window_size_right,
        bool has_softcap,
        int num_splits,
        std::optional<bool> pack_gqa_,
        int const sm_margin
);

/**
 *  Torch Library Registration
 */
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("fwd(Tensor!  q,"
            "    Tensor   k,"
            "    Tensor   v,"
            "    Tensor?  k_new,"
            "    Tensor?  v_new,"
            "    Tensor?  q_v,"
            "    Tensor!? out,"
            "    Tensor?  cu_seqlens_q,"
            "    Tensor?  cu_seqlens_k,"
            "    Tensor?  cu_seqlens_k_new,"
            "    Tensor?  seqused_q,"
            "    Tensor?  seqused_k,"
            "    int?     max_seqlen_q,"
            "    int?     max_seqlen_k,"
            "    Tensor?  page_table,"
            "    Tensor?  kv_batch_idx,"
            "    Tensor?  leftpad_k,"
            "    Tensor?  rotary_cos,"
            "    Tensor?  rotary_sin,"
            "    Tensor?  q_descale,"
            "    Tensor?  k_descale,"
            "    Tensor?  v_descale,"
            "    float    softmax_scale,"
            "    bool     is_causal,"
            "    int      window_size_left,"
            "    int      window_size_right,"
            "    float    softcap,"
            "    bool     is_rotary_interleaved,"
            "    Tensor?  scheduler_metadata,"
            "    int      num_splits,"
            "    bool?    pack_gqa,"
            "    int      sm_margin) -> Tensor[]");
    ops.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));

    ops.def("get_scheduler_metadata("
            "    int      batch_size,"
            "    int      max_seqlen_q,"
            "    int      max_seqlen_k,"
            "    int      num_heads,"
            "    int      num_heads_k,"
            "    int      headdim,"
            "    int      headdim_v,"
            "    ScalarType qkv_dtype,"
            "    Tensor   seqused_k,"
            "    Tensor?  cu_seqlens_q,"
            "    Tensor?  cu_seqlens_k,"
            "    Tensor?  cu_seqlens_k_new,"
            "    Tensor?  seqused_q,"
            "    Tensor?  leftpad_k,"
            "    int?     page_size,"
            "    int      max_seqlen_k_new," // 0 means we're not appending new KV
            "    bool     is_causal,"
            "    int      window_size_left,"
            "    int      window_size_right,"
            "    bool     has_softcap,"
            "    int      num_splits,"
            "    bool?    pack_gqa,"
            "    int      sm_margin) -> Tensor");
   ops.impl("get_scheduler_metadata", torch::kCUDA, 
        make_pytorch_shim(&mha_fwd_get_scheduler_metadata));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME);