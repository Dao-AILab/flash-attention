#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"

#include "ln.h"

/*

Supported Type combinations:

input  residual   compute   weights   output
============================================
fp32     fp32      fp32      fp32      fp32
fp16     fp32      fp32      fp32      fp16
fp16     fp16      fp32      fp32      fp16
bf16     fp32      fp32      fp32      bf16
bf16     bf16      fp32      fp32      bf16
fp16     fp16      fp32      fp16      fp16
bf16     bf16      fp32      bf16      bf16

Remarks:
Output type = Input type
Compute always in FP32

*/

namespace layer_norm {

// Create registries and provide runtime versions of config hash functions.

FwdRegistry FWD_FUNCS;
BwdRegistry BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(torch::Dtype dtype){
    if( dtype == torch::kFloat16 ) {
        return TypeId<fp16>::Value;
    } else if( dtype == torch::kBFloat16 ) {
        return TypeId<bf16>::Value;
    } else if( dtype == torch::kFloat32 ) {
        return TypeId<fp32>::Value;
    } else {
        TORCH_CHECK(false, "Type not supported: ", dtype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype, torch::Dtype otype, torch::Dtype ctype, uint64_t hidden_size) {
    using namespace layer_norm;
    uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(rtype) << 4) | (get_type_id(otype) << 6) | (get_type_id(ctype) << 8);
    uint64_t launcher_key = (type_key << 32) | hidden_size;
    return launcher_key;
}

}  // namespace layer_norm

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction & get_fwd_launcher(torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype, torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
    auto iter = layer_norm::FWD_FUNCS.find(layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != layer_norm::FWD_FUNCS.end() ) {
        return iter->second;
    } else {
        TORCH_CHECK(false, "FWD: Unsupported hidden_size or types: ", hidden_size, wtype, itype, rtype, otype, ctype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction & get_bwd_launcher(torch::Dtype wtype, torch::Dtype itype, torch::Dtype rtype, torch::Dtype otype, torch::Dtype ctype, uint32_t hidden_size) {
    auto iter = layer_norm::BWD_FUNCS.find(layer_norm::get_key(wtype, itype, rtype, otype, ctype, hidden_size));
    if( iter != layer_norm::BWD_FUNCS.end() ) {
        return iter->second;
    } else {
        TORCH_CHECK(false, "BWD: Unsupported hidden_size or types: ", hidden_size, wtype, itype, rtype, otype, ctype);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> dropout_add_ln_fwd(const at::Tensor &x0,      // Input: BxSxhidden_size
                                           c10::optional<const at::Tensor> &x1_,      // Residual: BxSxhidden_size
                                           const at::Tensor &gamma,   // hidden_size
                                           const at::Tensor &beta,   // hidden_size
                                           c10::optional<const at::Tensor> &rowscale_,      // BxS
                                           const float dropout_p,
                                           const float epsilon,
                                           c10::optional<at::Generator> gen_,
                                           bool residual_in_fp32
) {
    auto itype = x0.scalar_type();
    auto rtype = x1_.has_value()
        ? x1_.value().scalar_type()
        : (residual_in_fp32 ? torch::kFloat32 : x0.scalar_type());
    auto wtype = gamma.scalar_type();
    auto otype = itype;
    auto ctype = torch::kFloat32;
    auto mtype = torch::kUInt8;

    TORCH_CHECK(beta.scalar_type() == wtype);

    TORCH_CHECK(x0.is_cuda())
    TORCH_CHECK(gamma.is_cuda())
    TORCH_CHECK(beta.is_cuda())

    TORCH_CHECK(x0.is_contiguous());
    auto sizes = x0.sizes();
    TORCH_CHECK(sizes.size() == 2);

    const int rows = sizes[0];
    const int cols = sizes[1];
    auto hidden_size = gamma.numel();

    if (x1_.has_value()) {
        auto x1 = x1_.value();
        TORCH_CHECK(x1.is_cuda())
        TORCH_CHECK(x1.is_contiguous());
        TORCH_CHECK(x1.sizes() == sizes);
    }

    if (rowscale_.has_value()) {
        auto rowscale = rowscale_.value();
        TORCH_CHECK(rowscale.is_cuda())
        TORCH_CHECK(rowscale.is_contiguous());
        TORCH_CHECK(rowscale.sizes() == std::vector<int64_t>{rows});
        TORCH_CHECK(rowscale.scalar_type() == itype);
    }

    TORCH_CHECK(gamma.sizes() == beta.sizes());
    TORCH_CHECK(hidden_size == cols);

    TORCH_CHECK(epsilon >= 0.f);

    auto opts = x0.options();

    bool save_x = x1_.has_value() || (dropout_p > 0.f) || (itype != rtype);
    at::Tensor x;
    if (save_x) { x = torch::empty(sizes, opts.dtype(rtype)); }
    at::Tensor dmask;
    if (dropout_p > 0.f) { dmask = torch::empty(sizes, opts.dtype(mtype)); };
    auto z = torch::empty(sizes, opts.dtype(otype));

    auto mu = torch::empty({ rows }, opts.dtype(ctype));
    auto rsigma = torch::empty({ rows }, opts.dtype(ctype));

    layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

    launch_params.props = at::cuda::getCurrentDeviceProperties();
    launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.x1 = x1_.has_value() ? x1_.value().data_ptr() : nullptr;
    launch_params.params.rowscale = rowscale_.has_value() ? rowscale_.value().data_ptr() : nullptr;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    // Request the kernel launcher.
    auto launcher = get_fwd_launcher(wtype, itype, rtype, otype, ctype, hidden_size);

    // Query the kernel-specific launch parameters.
    launcher(launch_params, true);

    at::Tensor workspace, barrier;

    // Set the kernel runtime parameters.
    layer_norm::FwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x0 = x0.data_ptr();
    params.x = save_x ? x.data_ptr() : nullptr;
    params.dmask = dropout_p > 0.f ? dmask.data_ptr() : nullptr;
    params.mu = mu.data_ptr();
    params.rs = rsigma.data_ptr();
    params.gamma = gamma.data_ptr();
    params.beta = beta.data_ptr();
    params.z = z.data_ptr();
    params.epsilon = epsilon;
    params.dropout_scale = 1.f / (1.f - dropout_p);

    if (dropout_p > 0.f) {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        int64_t counter_offset = launch_params.elts_per_thread;

        // See Note [Acquire lock when using random generators]
        {
            std::lock_guard<std::mutex> lock(gen->mutex_);
            params.philox_args = gen->philox_cuda_state(counter_offset);
        }
    }

    if( launch_params.barrier_size > 0 ) {
        auto options = x0.options();
        barrier = torch::zeros(launch_params.barrier_size, options.dtype(torch::kInt32));
        workspace = torch::empty(launch_params.workspace_bytes, options.dtype(torch::kChar));
        params.workspace = workspace.data_ptr();
        params.barrier = barrier.data_ptr<int>();
    }

    // Launch the kernel.
    launcher(launch_params, false);

    return { z, x, dmask, mu, rsigma };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> dropout_add_ln_bwd(const at::Tensor &dz,     // BxSxhidden_size
                                           const at::Tensor &x,      // BxSxhidden_size
                                           c10::optional<const at::Tensor> &dmask_,  // BxSxhidden_size
                                           const at::Tensor &mu,     // BxS, FP32!
                                           const at::Tensor &rsigma, // BxS, FP32!
                                           const at::Tensor &gamma,   // hidden_size
                                           c10::optional<const at::Tensor> &rowscale_,      // BxS
                                           const float dropout_p,
                                           const bool has_residual
) {

    auto itype = dz.scalar_type();
    auto rtype = x.scalar_type();
    auto wtype = gamma.scalar_type();
    auto otype = itype;
    auto ctype = torch::kFloat32;
    auto mtype = torch::kUInt8;

    if (dropout_p > 0.f) { TORCH_CHECK(dmask_.has_value()); }

    TORCH_CHECK(dz.dtype() == otype);
    TORCH_CHECK(mu.dtype() == ctype);
    TORCH_CHECK(rsigma.dtype() == ctype);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(dz.is_cuda());
    TORCH_CHECK(mu.is_cuda());
    TORCH_CHECK(rsigma.is_cuda());
    TORCH_CHECK(gamma.is_cuda());

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(dz.is_contiguous());

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 2);
    TORCH_CHECK(dz.sizes() == sizes);
    auto rows = sizes[0];
    auto cols = sizes[1];

    if (dmask_.has_value()) {
        auto dmask = dmask_.value();
        TORCH_CHECK(dmask.dtype() == mtype);
        TORCH_CHECK(dmask.is_cuda());
        TORCH_CHECK(dmask.is_contiguous());
        TORCH_CHECK(dmask.sizes() == sizes);
    }

    if (rowscale_.has_value()) {
        auto rowscale = rowscale_.value();
        TORCH_CHECK(rowscale.is_cuda())
        TORCH_CHECK(rowscale.is_contiguous());
        TORCH_CHECK(rowscale.sizes() == std::vector<int64_t>{rows});
        TORCH_CHECK(rowscale.scalar_type() == itype);
    }

    auto hidden_size = gamma.numel();

    TORCH_CHECK(mu.numel() == rows);
    TORCH_CHECK(mu.sizes() == rsigma.sizes());

    TORCH_CHECK(gamma.numel() == cols);

    auto opts = x.options();

    auto dx0 = torch::empty_like(x, opts.dtype(itype));
    at::Tensor dx1;
    if (has_residual) { dx1 = torch::empty_like(x, opts.dtype(rtype)); }
    auto dgamma = torch::empty_like(gamma);
    auto dbeta = torch::empty_like(gamma);

    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
    launch_params.props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.dx1 = has_residual ? dx1.data_ptr() : nullptr;
    launch_params.params.rowscale = rowscale_.has_value() ? rowscale_.value().data_ptr() : nullptr;

    auto launcher = get_bwd_launcher(wtype, itype, rtype, otype, ctype, hidden_size);

    launcher(launch_params, true, /*prenorm=*/false);

    auto dgamma_part = torch::empty({ launch_params.params.ctas_per_col, hidden_size }, opts.dtype(ctype));
    auto dbeta_part = torch::empty({ launch_params.params.ctas_per_col, hidden_size }, opts.dtype(ctype));
    at::Tensor workspace, barrier;

    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x.data_ptr();
    params.dmask = dropout_p > 0.f ? dmask_.value().data_ptr() : nullptr;
    params.mu = mu.data_ptr();
    params.rs = rsigma.data_ptr();
    params.gamma = gamma.data_ptr();
    params.dz = dz.data_ptr();
    params.dx0 = dx0.data_ptr();
    params.dbeta = dbeta.data_ptr();
    params.dgamma = dgamma.data_ptr();
    params.dbeta_part = dbeta_part.data_ptr();
    params.dgamma_part = dgamma_part.data_ptr();
    params.dropout_scale = 1.f / (1.f - dropout_p);

    if( launch_params.barrier_size > 0 ) {
        // TODO Any way to avoid this?
        barrier = torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
        workspace = torch::empty(launch_params.workspace_bytes, opts.dtype(torch::kChar));
        params.workspace = workspace.data_ptr();
        params.barrier = barrier.data_ptr<int>();
    }

    launcher(launch_params, false, /*prenorm=*/false);

    return { dx0, dx1, dgamma, dbeta, dgamma_part, dbeta_part };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> dropout_add_ln_prenorm_bwd(const at::Tensor &dz,     // BxSxhidden_size
                                                   const at::Tensor &dx,     // BxSxhidden_size
                                                   const at::Tensor &x,      // BxSxhidden_size
                                                   c10::optional<const at::Tensor> &dmask_,  // BxSxhidden_size
                                                   const at::Tensor &mu,     // BxS, FP32!
                                                   const at::Tensor &rsigma, // BxS, FP32!
                                                   const at::Tensor &gamma,   // hidden_size
                                                   c10::optional<const at::Tensor> &rowscale_,      // BxS
                                                   const float dropout_p,
                                                   const bool has_residual
) {

    auto itype = dz.scalar_type();
    auto rtype = x.scalar_type();
    auto wtype = gamma.scalar_type();
    auto otype = itype;
    auto ctype = torch::kFloat32;
    auto mtype = torch::kUInt8;

    if (dropout_p > 0.f) { TORCH_CHECK(dmask_.has_value()); }

    TORCH_CHECK(dz.dtype() == otype);
    TORCH_CHECK(dx.dtype() == rtype);
    TORCH_CHECK(mu.dtype() == ctype);
    TORCH_CHECK(rsigma.dtype() == ctype);

    TORCH_CHECK(x.is_cuda());
    TORCH_CHECK(dz.is_cuda());
    TORCH_CHECK(dx.is_cuda());
    TORCH_CHECK(mu.is_cuda());
    TORCH_CHECK(rsigma.is_cuda());
    TORCH_CHECK(gamma.is_cuda());

    TORCH_CHECK(x.is_contiguous());
    TORCH_CHECK(dz.is_contiguous());
    TORCH_CHECK(dx.is_contiguous());

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 2);
    TORCH_CHECK(dz.sizes() == sizes);
    TORCH_CHECK(dx.sizes() == sizes);
    auto rows = sizes[0];
    auto cols = sizes[1];

    if (dmask_.has_value()) {
        auto dmask = dmask_.value();
        TORCH_CHECK(dmask.dtype() == mtype);
        TORCH_CHECK(dmask.is_cuda());
        TORCH_CHECK(dmask.is_contiguous());
        TORCH_CHECK(dmask.sizes() == sizes);
    }

    if (rowscale_.has_value()) {
        auto rowscale = rowscale_.value();
        TORCH_CHECK(rowscale.is_cuda())
        TORCH_CHECK(rowscale.is_contiguous());
        TORCH_CHECK(rowscale.sizes() == std::vector<int64_t>{rows});
        TORCH_CHECK(rowscale.scalar_type() == itype);
    }

    auto hidden_size = gamma.numel();

    TORCH_CHECK(mu.numel() == rows);
    TORCH_CHECK(mu.sizes() == rsigma.sizes());

    TORCH_CHECK(gamma.numel() == cols);

    auto opts = x.options();

    auto dx0 = torch::empty_like(x, opts.dtype(itype));
    at::Tensor dx1;
    if (has_residual) { dx1 = torch::empty_like(x, opts.dtype(rtype)); }
    auto dgamma = torch::empty_like(gamma);
    auto dbeta = torch::empty_like(gamma);

    layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
    launch_params.stream = at::cuda::getCurrentCUDAStream().stream();
    launch_params.props = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dropout_p < 1.f);
    launch_params.params.dropout_keep_p = 1.f - dropout_p;
    launch_params.params.dx1 = has_residual ? dx1.data_ptr() : nullptr;
    launch_params.params.rowscale = rowscale_.has_value() ? rowscale_.value().data_ptr() : nullptr;

    // TODO: how to set template param for launcher
    auto launcher = get_bwd_launcher(wtype, itype, rtype, otype, ctype, hidden_size);

    launcher(launch_params, true, /*prenorm=*/true);

    auto dgamma_part = torch::empty({ launch_params.params.ctas_per_col, hidden_size }, opts.dtype(ctype));
    auto dbeta_part = torch::empty({ launch_params.params.ctas_per_col, hidden_size }, opts.dtype(ctype));
    at::Tensor workspace, barrier;

    layer_norm::BwdParams &params = launch_params.params;
    params.rows = rows;
    params.cols = cols;
    params.x = x.data_ptr();
    params.dmask = dropout_p > 0.f ? dmask_.value().data_ptr() : nullptr;
    params.mu = mu.data_ptr();
    params.rs = rsigma.data_ptr();
    params.gamma = gamma.data_ptr();
    params.dz = dz.data_ptr();
    params.dx = dx.data_ptr();
    params.dx0 = dx0.data_ptr();
    params.dbeta = dbeta.data_ptr();
    params.dgamma = dgamma.data_ptr();
    params.dbeta_part = dbeta_part.data_ptr();
    params.dgamma_part = dgamma_part.data_ptr();
    params.dropout_scale = 1.f / (1.f - dropout_p);

    if( launch_params.barrier_size > 0 ) {
        // TODO Any way to avoid this?
        barrier = torch::zeros(launch_params.barrier_size, opts.dtype(torch::kInt32));
        workspace = torch::empty(launch_params.workspace_bytes, opts.dtype(torch::kChar));
        params.workspace = workspace.data_ptr();
        params.barrier = barrier.data_ptr<int>();
    }

    launcher(launch_params, false, /*prenorm=*/true);

    return { dx0, dx1, dgamma, dbeta, dgamma_part, dbeta_part };
}
////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA DropoutAddLayerNorm";
  m.def("dropout_add_ln_fwd", &dropout_add_ln_fwd, "Run Dropout + Add + LayerNorm forward kernel");
  m.def("dropout_add_ln_bwd", &dropout_add_ln_bwd, "Run Dropout + Add + LayerNorm backward kernel");
  m.def("dropout_add_ln_prenorm_bwd", &dropout_add_ln_prenorm_bwd, "Run Dropout + Add + LayerNorm (PreNorm version) backward kernel");
}
