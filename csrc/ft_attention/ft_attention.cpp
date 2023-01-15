#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include <c10/cuda/CUDAGuard.h>


#include "decoder_masked_multihead_attention.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(TYPE, NAME, ...)                  \
  if (TYPE == at::ScalarType::Half) {                                      \
    using scalar_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else if (TYPE == at::ScalarType::BFloat16) {                           \
    using scalar_t = at::BFloat16;                                         \
    __VA_ARGS__();                                                         \
  } else if (TYPE == at::ScalarType::Float)  {                             \
    using scalar_t = float;                                                \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for type '", toString(TYPE), "'"); \
  }

template<typename T>
void masked_multihead_attention(const Masked_multihead_attention_params<T>& params,
                                const cudaStream_t& stream);

template<typename T>
void cross_multihead_attention(const Masked_multihead_attention_params<T>& params,
                               const cudaStream_t& stream);

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<at::Half> {
    using Type = uint16_t;
};

template<>
struct SATypeConverter<at::BFloat16> {
    using Type = __nv_bfloat16;
};

template <typename T>
void set_params(Masked_multihead_attention_params<T> &params,
                const size_t batch_size,
                const size_t nheads,
                const size_t memory_max_seqlen,
                const size_t headdim,
                const int timestep,
                const int rotary_embedding_dim,
                const bool neox_rotary_style,
                const int qkv_batch_stride,
                T *q_ptr,
                T *k_ptr,
                T *v_ptr,
                T *k_cache_ptr,
                T *v_cache_ptr,
                int *length_per_sample,
                T *out_ptr) {
    // Reset the parameters
    memset(&params, 0, sizeof(params));
    params.q = q_ptr;
    params.k = k_ptr;
    params.v = v_ptr;
    params.q_bias = nullptr;
    params.k_bias = nullptr;
    params.v_bias = nullptr;
    params.k_cache = k_cache_ptr;
    params.v_cache = v_cache_ptr;
    params.out = out_ptr;
    params.cache_indir = nullptr;
    params.stride = qkv_batch_stride;
    params.batch_size = batch_size;
    params.beam_width = 1;
    params.memory_max_len = memory_max_seqlen;
    params.num_heads = nheads;
    params.hidden_size_per_head = headdim;
    params.rotary_embedding_dim = rotary_embedding_dim;
    params.neox_rotary_style = neox_rotary_style;
    params.timestep = timestep;
    params.inv_sqrt_dh = 1.f / sqrt(float(headdim));
    params.total_padding_tokens = nullptr;
    params.masked_tokens = nullptr;
    params.prefix_prompt_lengths = nullptr;
    params.max_prefix_prompt_length = 0;
    params.relative_attention_bias = nullptr;
    params.relative_attention_bias_stride = 0;
    params.cross_attention_out = nullptr;
    params.max_decoder_seq_len = 0;
    params.is_return_cross_attentions = false;
    params.finished = nullptr;
    params.memory_length_per_sample = nullptr;
    params.length_per_sample = length_per_sample;
}

torch::Tensor single_query_attention(const torch::Tensor q,
                                     const torch::Tensor k,
                                     const torch::Tensor v,
                                     torch::Tensor k_cache,
                                     torch::Tensor v_cache,
                                     c10::optional<const torch::Tensor> length_per_sample_,
                                     const int timestep,
                                     const int rotary_embedding_dim = 0,
                                     const bool neox_rotary_style=true) {
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(k_cache); CHECK_DEVICE(v_cache);
    int batch_size = v_cache.size(0);
    int nheads = v_cache.size(1);
    int memory_max_seqlen = v_cache.size(2);
    int headdim = v_cache.size(3);
    CHECK_SHAPE(q, batch_size, nheads, headdim);
    CHECK_SHAPE(k, batch_size, nheads, headdim);
    CHECK_SHAPE(v, batch_size, nheads, headdim);
    CHECK_SHAPE(v_cache, batch_size, nheads, memory_max_seqlen, headdim);
    // k_cache shape: [B, H, Dh/x, L, x] where x=8 for fp16 and x=4 for fp32
    int packsize = k_cache.dtype() == torch::kFloat32 ? 4 : 8;
    CHECK_SHAPE(k_cache, batch_size, nheads, headdim / packsize, memory_max_seqlen, packsize);
    TORCH_CHECK(q.stride(2) == 1 && q.stride(1) == headdim);
    TORCH_CHECK(k.stride(2) == 1 && k.stride(1) == headdim);
    TORCH_CHECK(v.stride(2) == 1 && v.stride(1) == headdim);
    TORCH_CHECK(q.stride(0) == k.stride(0) && q.stride(0) == v.stride(0));
    CHECK_CONTIGUOUS(v_cache); CHECK_CONTIGUOUS(k_cache);

    if (length_per_sample_.has_value()) {
        auto length_per_sample = length_per_sample_.value();
        CHECK_DEVICE(length_per_sample);
        CHECK_SHAPE(length_per_sample, batch_size);
        CHECK_CONTIGUOUS(length_per_sample);
        TORCH_CHECK(length_per_sample.dtype() == torch::kInt32);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    torch::Tensor out = torch::empty_like(q);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(q.scalar_type(), "single_query_attention", [&] {
        using DataType = typename SATypeConverter<scalar_t>::Type;
        Masked_multihead_attention_params<DataType> params;
        set_params(params, batch_size, nheads, memory_max_seqlen, headdim, timestep,
                   rotary_embedding_dim, neox_rotary_style, q.stride(0),
                   reinterpret_cast<DataType*>(q.data_ptr()),
                   reinterpret_cast<DataType*>(k.data_ptr()),
                   reinterpret_cast<DataType*>(v.data_ptr()),
                   reinterpret_cast<DataType*>(k_cache.data_ptr()),
                   reinterpret_cast<DataType*>(v_cache.data_ptr()),
                   length_per_sample_.has_value()
                       ? length_per_sample_.value().data_ptr<int>() : nullptr,
                   reinterpret_cast<DataType*>(out.data_ptr()));
        auto stream = at::cuda::getCurrentCUDAStream();
        masked_multihead_attention(params, stream);
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("single_query_attention", &single_query_attention, "Attention with a single query",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("length_per_sample_"), py::arg("timestep"), py::arg("rotary_embedding_dim")=0,
          py::arg("neox_rotary_style")=true);
}
