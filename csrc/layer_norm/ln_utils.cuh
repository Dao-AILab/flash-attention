#pragma once

#include <cassert>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ln.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr uint32_t THREADS_PER_WARP = 32;

////////////////////////////////////////////////////////////////////////////////////////////////////

inline void check_cuda_(cudaError_t status, const char *file, int line) {
    if( status != cudaSuccess ) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(status), file, line);
        exit(status);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(ans)                                                                                                        \
    { check_cuda_((ans), __FILE__, __LINE__); }

////////////////////////////////////////////////////////////////////////////////////////////////////

#define DIVUP(x, y) (((x) + ((y)-1)) / (y))

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_FWD_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG)                 \
    void ln_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE(LaunchParams<FwdParams> &launch_params,                      \
                                                                                const bool configure_params) {                               \
        launch_<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>(                    \
            launch_params, configure_params);                                                                                                \
    }                                                                                                                                        \
    static FwdRegistrar<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, HIDDEN_SIZE> reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE( \
        ln_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_BWD_LAUNCHER(                                                                                                                  \
    HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINALIZE)                      \
    void ln_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE(LaunchParams<BwdParams> &launch_params,                         \
                                                                                const bool configure_params) {                                  \
        launch_<WTYPE,                                                                                                                          \
                ITYPE,                                                                                                                          \
                RTYPE,                                                                                                                          \
                OTYPE,                                                                                                                          \
                CTYPE,                                                                                                                          \
                uint32_t,                                                                                                                       \
                HIDDEN_SIZE,                                                                                                                    \
                CTAS_PER_ROW,                                                                                                                   \
                WARPS_M,                                                                                                                        \
                WARPS_N,                                                                                                                        \
                BYTES_PER_LDG,                                                                                                                  \
                BYTES_PER_LDG_FINALIZE>(launch_params, configure_params);                                                                       \
    }                                                                                                                                           \
    static BwdRegistrar<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, HIDDEN_SIZE> reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE(    \
        ln_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_PARALLEL_FWD_LAUNCHER(HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG)                \
    void ln_parallel_residual_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE(LaunchParams<FwdParams> &launch_params,            \
                                                                                const bool configure_params) {                                       \
        launch_parallel_residual_<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, uint32_t, HIDDEN_SIZE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>(          \
            launch_params, configure_params);                                                                                                        \
    }                                                                                                                                                \
    static FwdParallelRegistrar<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, HIDDEN_SIZE> reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE( \
        ln_parallel_residual_fwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define REGISTER_PARALLEL_BWD_LAUNCHER(                                                                                                              \
    HIDDEN_SIZE, WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG, BYTES_PER_LDG_FINALIZE)                           \
    void ln_parallel_residual_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE(LaunchParams<BwdParams> &launch_params,            \
                                                                                const bool configure_params) {                                       \
        launch_parallel_residual_<WTYPE,                                                                                                             \
                ITYPE,                                                                                                                               \
                RTYPE,                                                                                                                               \
                OTYPE,                                                                                                                               \
                CTYPE,                                                                                                                               \
                uint32_t,                                                                                                                            \
                HIDDEN_SIZE,                                                                                                                         \
                CTAS_PER_ROW,                                                                                                                        \
                WARPS_M,                                                                                                                             \
                WARPS_N,                                                                                                                             \
                BYTES_PER_LDG,                                                                                                                       \
                BYTES_PER_LDG_FINALIZE>(launch_params, configure_params);                                                                            \
    }                                                                                                                                                \
    static BwdParallelRegistrar<WTYPE, ITYPE, RTYPE, OTYPE, CTYPE, HIDDEN_SIZE> reg_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE( \
        ln_parallel_residual_bwd_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##RTYPE##_##OTYPE##_##CTYPE)

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 operator+(const float2 & a, const float2 & b){
    return {a.x + b.x, a.y + b.y};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void operator+=(float2 & a, const float2 & b){
    a.x += b.x;
    a.y += b.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Sum {
    inline __device__ Sum(){}
    inline __device__ T operator()(const T &a, const T &b){
        return a + b;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ T warp_shuffle_xor(const T & x, uint32_t idx){
    return __shfl_xor_sync(uint32_t(-1), x, idx);
}

template<>
inline __device__ float2 warp_shuffle_xor<float2>(const float2 & x, uint32_t idx){
    return { warp_shuffle_xor(x.x, idx), warp_shuffle_xor(x.y, idx) };
}

template<typename T>
inline __device__ T warp_shuffle_down(const T & x, uint32_t idx){
    return __shfl_down_sync(uint32_t(-1), x, idx);
}

template<>
inline __device__ float2 warp_shuffle_down<float2>(const float2 & x, uint32_t idx){
    return { warp_shuffle_down(x.x, idx), warp_shuffle_down(x.y, idx) };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace layer_norm {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint16 {
    uint4 u;
    uint4 v;
    uint4 s;
    uint4 t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct uint8 {
    uint4 u;
    uint4 v;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int BYTES>
struct BytesToType {};

template<>
struct BytesToType<64> {
    using Type = uint16;
    static_assert(sizeof(Type) == 64);
};

template<>
struct BytesToType<32> {
    using Type = uint8;
    static_assert(sizeof(Type) == 32);
};

template<>
struct BytesToType<16> {
    using Type = uint4;
    static_assert(sizeof(Type) == 16);
};

template<>
struct BytesToType<8> {
    using Type = uint64_t;
    static_assert(sizeof(Type) == 8);
};

template<>
struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};

template<>
struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};

template<>
struct BytesToType<1> {
    using Type = uint8_t;
    static_assert(sizeof(Type) == 1);
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct TypeToVec2 {};

template<>
struct TypeToVec2<float> {
    using Type = float2;
};

template<>
struct TypeToVec2<half> {
    using Type = half2;
};

template<>
struct TypeToVec2<nv_bfloat16> {
    using Type = nv_bfloat162;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int INDEX>
struct Get {
    template<typename T, typename R>
    static inline __device__ R of(const T &vec);
};

template<>
template<typename T, typename R>
inline __device__ R Get<0>::of(const T &vec) {
    return vec.x;
}

template<>
template<typename T, typename R>
inline __device__ R Get<1>::of(const T &vec) {
    return vec.y;
}

template<>
template<typename T, typename R>
inline __device__ R Get<2>::of(const T &vec) {
    return vec.z;
}

template<>
template<typename T, typename R>
inline __device__ R Get<3>::of(const T &vec) {
    return vec.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Src, typename Dst>
struct Converter{
    static inline __device__ Dst convert(const Src &from) {
        return Dst(from);
    }
};

template<>
struct Converter<float2, half2>{
    static inline __device__ half2 convert(const float2 &x) {
        return __float22half2_rn(x);
    }
};

template<>
struct Converter<float2, nv_bfloat162>{
    static inline __device__ nv_bfloat162 convert(const float2 &x) {
#if __CUDA_ARCH__ >= 800
        return __float22bfloat162_rn(x);
#else
        union {
            nv_bfloat162 raw;
            nv_bfloat16 x;
            nv_bfloat16 y;
        } tmp;
        tmp.x = __float2bfloat16_rn(x.x);
        tmp.y = __float2bfloat16_rn(x.y);
        return tmp.raw;
#endif
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Zeros{
    static inline __device__ T get() {
        return T(0.f);
    }
};

template<> 
struct Zeros<float2>{
    static inline __device__ float2 get() {
        return make_float2(0.f, 0.f);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Elt_type, uint32_t NUM_ELT>
struct Vec {

    enum { BYTES = NUM_ELT * sizeof(Elt_type) };

    using Vec_type = typename BytesToType<BYTES>::Type;

    using Alias_type = union {
        Vec_type vec;
        Elt_type elt[NUM_ELT];
    };

    Alias_type data;

    template<typename S>
    inline __device__ void to(Vec<S, NUM_ELT> &other) {
        #pragma unroll
        for( int it = 0; it < NUM_ELT; it++ ) {
            other.data.elt[it] = S(this->data.elt[it]);
        }
    }

    template<typename Op>
    inline __device__ void assign(const Op &op) {
        #pragma unroll
        for( int it = 0; it < NUM_ELT; it++ ) {
            this->data.elt[it] = op(it);
        }
    }

    inline __device__ void zero_() {
        #pragma unroll
        for( int it = 0; it < NUM_ELT; it++ ) {
            this->data.elt[it] = Elt_type(0.f);
        }
    }

    inline __device__ void load_from(const void *base_ptr, const size_t idx) {
        this->data.vec = static_cast<const Vec_type *>(base_ptr)[idx];
    }

    inline __device__ void store_to(void *base_ptr, const size_t idx) {
        static_cast<Vec_type *>(base_ptr)[idx] = this->data.vec;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<uint32_t CTAS_PER_ROW>
struct InterCTASync {

    template<typename Params>
    inline __device__ InterCTASync(Params & params, uint32_t bidm, uint32_t bidn)
        : phase_counter_(0)
        , b0_(params.barrier + bidm) // The barrier for this group of CTAs.
        , b1_(params.barrier + bidm + params.ctas_per_col) // The barrier for this group of CTAs.
    {
        // BARRIERS ARE ASSUMED TO BE INITIALIZED TO 0!
    }

    inline __device__ void spin_wait_(int *barrier, int step, int expected) {
        asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(barrier), "r"(step));
        for( int found = -1; found != expected; ) {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(found) : "l"(barrier));
        }
    }

    inline __device__ void sync(){
        // ALL THREADS MUST ENTER!

        // We switch barrier every iteration.
        int *barrier = phase_counter_ & 0x1 ? b1_ : b0_;
        // We decrement every other iteration.
        bool dec = phase_counter_ & 0x2;
        int step = dec ? -1 : 1;
        int expected = dec ? 0 : CTAS_PER_ROW;
        // There are only 4 phases: up/down for b0/b1.
        phase_counter_ = (phase_counter_ + 1) & 0x3;

        if( threadIdx.x == 0 ) {
            spin_wait_(barrier, step, expected);
        }
        // CTA waits for thread 0
        __syncthreads();
    }

    int phase_counter_;
    int * b0_;
    int * b1_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer : public Reducer<T, 1, WARPS_M, WARPS_N> {

    using InterCTASync = InterCTASync<CTAS_PER_ROW>;
    using Base = Reducer<T, 1, WARPS_M, WARPS_N>;
    using Type = typename Base::Type;

    enum { SMEM_BYTES = Base::SMEM_BYTES };

    enum { WS_BARRIER_BYTES = 2 * sizeof(int) };
    enum { WS_DATA_BYTES = WARPS_M * CTAS_PER_ROW * sizeof(T) };

    // size of the barriers + temporary result per CTA (multiply with CTAS_PER_ROW to get total)
    enum { WORKSPACE_BYTES_PER_GROUP = Base::WORKSPACE_BYTES_PER_GROUP + WS_BARRIER_BYTES + WS_DATA_BYTES };

    template<typename Params>
    inline __device__ Reducer(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem)
        : Base(params, bidm, bidn, warp_m, warp_n, lane, smem) 
        , inter_cta_(params, bidm, bidn)
        , bidn_(bidn) // CTA id within the group.
        , w0_(static_cast<T*>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW)
        , w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW)
    {
    }

    template<typename Op>
    inline __device__ T allreduce(T data, Op &op) {
        data = Base::reduce(data, op);
        // We switch workspace every iteration.
        T *workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

        // Warp leaders 0 hold the CTA-local results.
        if( this->warp_n_ == 0 && this->lane_ == 0 ) {
            workspace[bidn_] = data;
        }
        inter_cta_.sync();
        static_assert(CTAS_PER_ROW <= 32);
        T total = Zeros<T>::get();
        if(this->lane_ < CTAS_PER_ROW){
            total = workspace[this->lane_];
        }
        total = Reducer<T, 1, 1, 1>::allreduce_(total, op);

        return total;
    }

    InterCTASync inter_cta_;

    T *w0_;
    T *w1_;
    int bidn_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M>
struct Reducer<T, 1, WARPS_M, 1> {

    using Type = T;
    enum { SMEM_BYTES = 0 };
    enum { WORKSPACE_BYTES_PER_GROUP = 0 };

    enum { THREADS_PER_WARP = 32 };

    template<typename Params>
    inline __device__ Reducer(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem) 
        : warp_n_(warp_n)
        , lane_(lane)
    {
    }

    template<typename Op>
    static inline __device__ T allreduce_(T data, Op &op) {
        #pragma unroll
        for( int it = 1; it < THREADS_PER_WARP; it *= 2 ) {
            data = op(data, warp_shuffle_xor(data, it));
        }
        return data;
    }

    template<typename Op>
    inline __device__ T allreduce(T data, Op &op) {
        return allreduce_(data, op);
    }

    template<typename Op>
    inline __device__ T reduce(T data, Op &op){
        // only lane 0 holds the result!
        #pragma unroll
        for( int it = THREADS_PER_WARP / 2; it > 0; it /= 2 ) {
            data = op(data, warp_shuffle_down(data, it));
        }  
        return data;
    }
    int warp_n_;
    int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Reducer<T, 1, WARPS_M, WARPS_N> : public Reducer<T, 1, WARPS_M, 1> {

    using Base = Reducer<T, 1, WARPS_M, 1>;

    using Type = T;

    enum { SMEM_BYTES = Base::SMEM_BYTES + WARPS_M * WARPS_N * sizeof(T) * 2 };
    enum { WORKSPACE_BYTES_PER_GROUP = 0 };

    enum { THREADS_PER_WARP = 32 };

    template<typename Params>
    inline __device__ Reducer(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem) 
        : Base(params, bidm, bidn, warp_m, warp_n, lane, smem) 
        , use0_(true)
    {
        smem0_ = &static_cast<T *>(smem)[warp_m * WARPS_N];
        smem1_ = smem0_ + WARPS_M * WARPS_N;
    }

    template<typename Op>
    inline __device__ T allreduce(T data, Op & op) {
        T * smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        data = Base::reduce(data, op);
        if( this->lane_ == 0 ) {
            smem[this->warp_n_] = data;
        }
        __syncthreads();
        T out = Zeros<T>::get();
        #pragma unroll
        for( int it = 0; it < WARPS_N; it++ ) {
            out = op(out, smem[it]);
        }
        return out;
    }

    template<typename Op>
    inline __device__ T reduce(T data, Op &op) {
        T * smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        // only intra-CTA group leader holds the result!
        data = Base::reduce(data, op);
        if( this->lane_ == 0 ) {
            smem[this->warp_n_] = data;
        }
        __syncthreads();
        T out = Zeros<T>::get();
        if( this->warp_n_ == 0 && this->lane_ == 0 ) {
            #pragma unroll
            for( int it = 0; it < WARPS_N; it++ ) {
                out = op(out, smem[it]);
            }
        }
        return out;
    }

    T * smem0_;
    T * smem1_;
    bool use0_;

};

////////////////////////////////////////////////////////////////////////////////////////////////////
 
template<typename T, typename int_t>
inline __device__ void warp_chan_upd_dynamic(T &m_a, T &m2_a, int_t &n_a, int num_active){
    //Assume at least leftmost is valid and init: step = next_pow2(num_active) / 2 (might get NaN otherwise)
    const int highest_bit_set = (8 * sizeof(num_active)) - __clz(num_active - 1);
    
    #pragma unroll
    for( int step = (1 << (highest_bit_set - 1)); step > 0; step /= 2 ) {
        // Exchange
        int_t n_b = warp_shuffle_down(n_a, step);
        T m_b = warp_shuffle_down(m_a, step);
        T m2_b = warp_shuffle_down(m2_a, step);

        // Update
        const int_t n_ab = n_a + n_b; // We can handle one of them being 0, not both.
        const T rn_ab = 1.f / n_ab; // Might have different n per thread, otherwise this would simplify :(
        const T delta = m_a - m_b;
        const float m2_ab = m2_a + m2_b + delta * delta * n_a * n_b * rn_ab;
        const float m_ab = (n_a * m_a + n_b * m_b) * rn_ab;

        n_a = n_ab;
        m_a = m_ab;
        m2_a = m2_ab;
    }
    // Intra-warp broadcast (only lane 0 has valid stats).
    m_a = __shfl_sync(uint32_t(-1), m_a, 0);
    m2_a = __shfl_sync(uint32_t(-1), m2_a, 0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t CTAS_PER_ROW, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats {
    // This could be done generically with the Reducer. But then we would have to exchange 3 instead of 2 fields.

    using InterCTASync = InterCTASync<CTAS_PER_ROW>;
    using BlockStats = Stats<T, 1, WARPS_M, WARPS_N>;
    using stats_t = typename BlockStats::stats_t;

    enum { SMEM_BYTES = BlockStats::SMEM_BYTES };

    template<typename Params>
    inline __device__ Stats(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem) 
        : inter_cta_(params, bidm, bidn)
        , block_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , bidn_(bidn) // CTA id within the group.
        , w0_(static_cast<stats_t*>(params.workspace) + (bidm * WARPS_M + warp_m) * CTAS_PER_ROW)
        , w1_(w0_ + params.ctas_per_col * WARPS_M * CTAS_PER_ROW)
        , warp_n_(warp_n)
        , lane_(lane)
    {
    }

    template<uint32_t N>
    inline __device__ stats_t compute(const T (&elts)[N], const T rn) {
        constexpr T ELTS_PER_ROW_PER_CTA = N * WARPS_N * THREADS_PER_WARP;
        // TODO rn is not really needed here..
        constexpr T block_rn = 1.f / T(ELTS_PER_ROW_PER_CTA);
        stats_t block_stats = block_stats_.compute(elts, block_rn);

        stats_t *workspace = inter_cta_.phase_counter_ & 0x1 ? w1_ : w0_;

        if( warp_n_ == 0 && lane_ == 0 ) {
            workspace[bidn_] = block_stats;
        }

        // Wait for all CTAS_PER_ROW CTAS in the group to have written their result.
        inter_cta_.sync();

        T n = Zeros<T>::get();
        T m = Zeros<T>::get();
        T m2 = Zeros<T>::get();

        // Assume CTA group size in N less than 32, such that we can finalize with a single warp.
        static_assert(CTAS_PER_ROW <= 32);

        // Every warp does the final reduction locally. 
        if( lane_ < CTAS_PER_ROW ) {
            stats_t result = workspace[lane_];
            n = ELTS_PER_ROW_PER_CTA;
            m = layer_norm::Get<0>::of<stats_t, T>(result);
            m2 = layer_norm::Get<1>::of<stats_t, T>(result);
        }

        warp_chan_upd_dynamic(m, m2, n, CTAS_PER_ROW);

        return { m, m2 };
    }

    InterCTASync inter_cta_;
    BlockStats block_stats_;

    stats_t *w0_;
    stats_t *w1_;
    int bidn_;
    int warp_n_;
    int lane_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M, uint32_t WARPS_N>
struct Stats<T, 1, WARPS_M, WARPS_N> {

    using WarpStats = Stats<T, 1, WARPS_M, 1>;
    using stats_t = typename WarpStats::stats_t;

    enum { SMEM_BYTES = WARPS_M * WARPS_N * sizeof(stats_t) * 2 };

    template<typename Params>
    inline __device__ Stats(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem) 
        : warp_stats_(params, bidm, bidn, warp_m, warp_n, lane, smem)
        , use0_(true)
    {
        smem0_ = static_cast<stats_t*>(smem) + warp_m * WARPS_N;
        smem1_ = smem0_ + WARPS_M * WARPS_N;
    }

    template<bool Is_even_cols, uint32_t N, typename function_t>
    inline __device__ stats_t compute(const T (&elts)[N], const T row_norm_factor,
                                      function_t valid_elts_in_warp_fn, const int num_valid_elts = N) {
        stats_t * smem = use0_ ? smem0_ : smem1_;
        use0_ = !use0_;
        // Compute warp local for all WARPS_N
        const auto warp_n = warp_stats_.reducer_.warp_n_;
        const T warp_norm_factor = 1.f / T(Is_even_cols ? N * THREADS_PER_WARP : valid_elts_in_warp_fn(warp_n));
        stats_t warp_stats = warp_stats_.template compute<Is_even_cols>(
            elts, warp_norm_factor, valid_elts_in_warp_fn, num_valid_elts
        );

        //Each warp warp leader stores its stats
        const auto lane = warp_stats_.reducer_.lane_;
        if( lane == 0 ) {
            smem[warp_n] = warp_stats;
        }
        __syncthreads();

        int n = 0;;
        T m = Zeros<T>::get();
        T m2 = Zeros<T>::get();

        // Assume that there are less than 32 warps, such that we can finalize with a single warp
        static_assert(WARPS_N <= 32);
        if(lane < WARPS_N){
            stats_t result = smem[lane];
            n = Is_even_cols ? N * THREADS_PER_WARP : valid_elts_in_warp_fn(lane);
            m = layer_norm::Get<0>::of<stats_t, T>(result);
            m2 = layer_norm::Get<1>::of<stats_t, T>(result);
        }

        warp_chan_upd_dynamic(m, m2, n, WARPS_N);

        return { m, m2 };
    }
    WarpStats warp_stats_;
    stats_t * smem0_;
    stats_t * smem1_;
    bool use0_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, uint32_t WARPS_M>
struct Stats<T, 1, WARPS_M, 1> {

    using stats_t = typename TypeToVec2<T>::Type;
    // The simple Warp reducer.
    using Reducer = Reducer<T, 1, WARPS_M, 1>;

    enum { SMEM_BYTES = 0 };

    template<typename Params>
    inline __device__ Stats(Params & params, uint32_t bidm, uint32_t bidn, uint32_t warp_m, uint32_t warp_n, uint32_t lane, void * smem) 
        : reducer_(params, bidm, bidn, warp_m, warp_n, lane, smem)
    {
    }

    template<bool Is_even_cols, uint32_t N, typename function_t>
    inline __device__ stats_t compute(const T (&elts)[N], const T row_norm_factor,
                                      // const int valid_elts_in_warp_ignored_, const int num_valid_elts = N) {
                                      function_t valid_elts_in_warp_fn, const int num_valid_elts = N) {

        auto sum = Sum<T>();

        T m = Zeros<T>::get();
        #pragma unroll
        for( int it = 0; it < N; it++ ) {
            if (Is_even_cols || (it < num_valid_elts)) {
                m += elts[it];
            }
        }
        m = reducer_.allreduce(m, sum) * row_norm_factor;

        T m2 = Zeros<T>::get();
        #pragma unroll
        for( int it = 0; it < N; it++ ) {
            if (Is_even_cols || (it < num_valid_elts)) {
                T diff = (elts[it] - m);
                m2 += diff * diff;
            }
        }
        m2 = reducer_.allreduce(m2, sum);

        return {m, m2};
    }

    Reducer reducer_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace layer_norm
