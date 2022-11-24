#include<ATen/ATen.h>

#include "fmha_api.cpp"
#include "src/fmha_fprop_fp16_kernel.gfx90a.cpp"


int main(){
    //int head_size = 64;
    int batch_size = 64;
    int nheads = 16
    int seqlen = 256
    int n = 1024
    int d = n / nheads; //head_size

    //initialize the tensors
    at::Tensor q = at::rand({batch_size*seqlen, nheads, d},at::kHalf) ;
    at::Tensor k = at::rand({batch_size*seqlen, nheads, d},at::kHalf) ;
    at::Tensor v = at::rand({batch_size*seqlen, nheads, d},at::kHalf) ;
    at::Tensor out = at::zeros({batch_size*seqlen, nheads, d},at::kHalf) ;

    at::Tensor cu_seqlens_q = at::full({batch_size + 1}, seqlen);
    at::Tensor cu_seqlens_k = at::full({batch_size + 1}, seqlen);

    int max_seqlen_q_ = 256;
    int max_seqlen_k_ = 256;
    
    //option parameters
    float p_dropout = 0;
    float softmax_scale = 0.125;
    bool zero_tensors = false;
    bool is_causal = false;
    bool return_softmax = false;
    int num_splits = 0;

    auto result =
    mha_fwd(const at::Tensor &q,         // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
            const at::Tensor &k,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            const at::Tensor &v,         // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            at::Tensor &out,             // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
            const at::Tensor &cu_seqlens_q,  // b+1
            const at::Tensor &cu_seqlens_k,  // b+1
            const int max_seqlen_q_,
            const int max_seqlen_k_,
            const float p_dropout,
            const float softmax_scale,
            const bool zero_tensors,
            const bool is_causal,
            const bool return_softmax,
            const int num_splits,
            c10::optional<at::Generator> gen_)


    return 0;
}