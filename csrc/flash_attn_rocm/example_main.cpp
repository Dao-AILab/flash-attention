#include<ATen/ATen.h>

#include "fmha_api.cpp"
#include "src/fmha_fprop_fp16_kernel.gfx90a.cpp"


int main(){

    int batch_size = 64;
    int nheads = 16;
    int seqlen = 256;
    int n = 1024;
    int d = n / nheads; //head_size

    //initialize the tensors
    at::Tensor q = at::rand({batch_size*seqlen, nheads, d},at::kHalf).to(at::kCUDA);
    at::Tensor k = at::rand({batch_size*seqlen, nheads, d},at::kHalf).to(at::kCUDA);
    at::Tensor v = at::rand({batch_size*seqlen, nheads, d},at::kHalf).to(at::kCUDA);
    //initialize the output tensor
    at::Tensor out = at::zeros({batch_size*seqlen, nheads, d},at::kHalf).to(at::kCUDA);

    //initialize seqlens vector (size is b+1)
    std::vector<int> cu_seqlens_q_vec;
    std::vector<int> cu_seqlens_k_vec;

    for (int i = 0 ; i < batch_size + 1; i++){
      cu_seqlens_q_vec.push_back(i * seqlen);
      cu_seqlens_k_vec.push_back(i * seqlen);
    }

    at::TensorOptions opts=at::TensorOptions().dtype(at::kInt);
    c10::IntArrayRef s={batch_size + 1};
    at::Tensor cu_seqlens_q=at::from_blob(cu_seqlens_q_vec.data(),s,opts).clone().to(at::kCUDA);
    at::Tensor cu_seqlens_k=at::from_blob(cu_seqlens_k_vec.data(),s,opts).clone().to(at::kCUDA);

    int max_seqlen_q_ = 256;
    int max_seqlen_k_ = 256;
    
    //option parameters
    float p_dropout = 0;          //dropout pecentage 
    float softmax_scale = 0.125;  //scale parameter
    bool zero_tensors = false;    //if init the out tensor into zeros
    bool is_causal = false;       //if do uptriangle mask
    bool return_softmax = false;  //if return the Intermediate results of softmax
    int num_splits = 0;           //parameter used in CUDA flash-attention, useless in ck

    //call the API and return results
    auto result = 
    mha_fwd(q,   
            k,   
            v,   
            out, 
            cu_seqlens_q, 
            cu_seqlens_k, 
            max_seqlen_q_,
            max_seqlen_k_,
            p_dropout,
            softmax_scale,
            zero_tensors,
            is_causal,
            return_softmax,
            num_splits,
            c10::optional<at::Generator> gen_);

    return 0;
}