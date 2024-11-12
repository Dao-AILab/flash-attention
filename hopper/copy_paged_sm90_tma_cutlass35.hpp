
#pragma once

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/version.h>

static_assert(CUTLASS_VERSION < 360, "CUTLASS 3.5.x is required for this file due to incompatible API changes in Cutlass. Cutlass 3.5 does not have the cache_hint argument to SM90_TMA_LOAD ops.");


struct PagedCopyArgs {

  CUTE_HOST_DEVICE
  PagedCopyArgs() : block_table_batch_stride{0}, page_block_size(0), block_table(nullptr)  {
  };

  CUTE_HOST_DEVICE
  PagedCopyArgs(int64_t const block_table_batch_stride_, int const page_block_size_, int32_t *block_table_) : block_table_batch_stride{block_table_batch_stride_}, page_block_size(page_block_size_), block_table(block_table_)  {
  };

  int64_t block_table_batch_stride; // The stride between block tables for different batches
  int page_block_size; // The size of a page block in number of elements
  int32_t* block_table; // The block table, must be properly sized or a nullptr
};

namespace cute {

  struct SM90_TMA_LOAD_PAGED
  {
    using COPY_OP = SM90_TMA_LOAD; // The underlying copy operation that we delegate work to

    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, uint64_t* mbar_ptr,
        void      * smem_ptr,
        int32_t const& crd0)
    {
      CUTE_INVALID_CONTROL_PATH("PAGED_COPY_OP not implemented for 1D");
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, uint64_t* mbar_ptr,
        PagedCopyArgs const& pca,
        void      * smem_ptr,
        int32_t const& crd0, int32_t const& crd1)
    {
      CUTE_INVALID_CONTROL_PATH("PAGED_COPY_OP not implemented for 2D");
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, uint64_t* mbar_ptr,
        PagedCopyArgs const& pca,
        void      * smem_ptr,
        int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
    {
      if (pca.block_table == nullptr) {
        return SM90_TMA_LOAD_3D::copy(desc_ptr, mbar_ptr, smem_ptr, crd0, crd1, crd2);
      }
      CUTE_INVALID_CONTROL_PATH("PAGED_COPY_OP not implemented for 3D");
    }
    CUTE_HOST_DEVICE static void
    copy(void const* desc_ptr, uint64_t* mbar_ptr,
        PagedCopyArgs const& pca,
        void      * smem_ptr,
       // Index order reordered for TMA from PagedSeqLenTraits::get_kv_gmem_layout()
       // via cute::make_tma_copy_atom ( see detail::construct_tma_gbasis )
       // and detail::make_tma_copy_desc to create a TMA descriptor.
       // The same reordering is aplied prior to calling via cute::tma_partition.

       // Final order determined experimentally.
       int32_t const& crdK, // embedding dim
       int32_t const& crdM, // sequence dim
       int32_t const& crdH, // head dim
       int32_t const& crdB) // batch dim
  {
    //auto log = pca.debug_log->nextline();
    //log.append_threadinfo();
    //log.snprintf("SM_90_TMA_LOAD_PAGED::copy(%d, %d, %d, %d) ", (int)crdM, (int)crdK, (int)crdH, (int)crdB);
    if (pca.block_table == nullptr) {
        return SM90_TMA_LOAD_4D::copy(desc_ptr, mbar_ptr, smem_ptr, crdK, crdM, crdH, crdB);
    }
    int32_t const page_idx_offset = crdM / pca.page_block_size; // page index within the batch entry
    int32_t const seq_pos_offset = crdM - page_idx_offset*pca.page_block_size; // == crd1 % page_block_size_ -> sequence position within the page
    int32_t const page_idx = pca.block_table[page_idx_offset + crdB*pca.block_table_batch_stride]; // The page index for the given batch and sequence position
    //if (cute::thread0()) {
    //  printf("SM90_TMA_LOAD_PAGED::copy crdM=%d, crdB=%d, crdK=%d, crdH=%d, page_idx=%d, seq_pos_offset=%d, ptr=%p\n", (int)crdM, (int)crdB, (int) crdK, (int) crdH, (int)page_idx, (int)seq_pos_offset, (void*)desc_ptr);
    //}
    return SM90_TMA_LOAD_4D::copy(desc_ptr, mbar_ptr, smem_ptr, crdK, seq_pos_offset, crdH, page_idx);
  }


  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr,
      void      * smem_ptr,
      int32_t const& crd0, int32_t const& crd1, int32_t const& crd2, int32_t const& crd3, int32_t const& crd4)
  {
    CUTE_INVALID_CONTROL_PATH("PAGED_COPY_OP not implemented for 5D");
  }

  };

struct SM90_TMA_LOAD_MULTICAST_PAGED
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       void      * smem_ptr,
       int32_t const& crd0)
  {
    CUTE_INVALID_CONTROL_PATH("not implemented");
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       PagedCopyArgs const& pca,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
    CUTE_INVALID_CONTROL_PATH("not implemented");
  }
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       PagedCopyArgs const& pca,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1, int32_t const& crd2)
   {
      if (pca.block_table == nullptr) {
        return SM90_TMA_LOAD_MULTICAST_3D::copy(desc_ptr, mbar_ptr, multicast_mask, smem_ptr, crd0, crd1, crd2);
      }
      CUTE_INVALID_CONTROL_PATH("PAGED_COPY_OP not implemented for 3D");
    }


  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint16_t multicast_mask,
       PagedCopyArgs const& pca,
       void      * smem_ptr,
       // Index order reordered for TMA from PagedSeqLenTraits::get_kv_gmem_layout()
       // via cute::make_tma_copy_atom ( see detail::construct_tma_gbasis )
       // and detail::make_tma_copy_desc to create a TMA descriptor.
       // The same reordering is aplied prior to calling via cute::tma_partition.

       // Final order determined experimentally.
       int32_t const& crdK, // embedding dim
       int32_t const& crdM, // sequence dim
       int32_t const& crdH, // head dim
       int32_t const& crdB) // batch dim
  {
    if (pca.block_table == nullptr) {
        return SM90_TMA_LOAD_MULTICAST_4D::copy(desc_ptr, mbar_ptr, multicast_mask, smem_ptr, crdK, crdM, crdH, crdB);
    }
    int32_t const page_idx_offset = crdM / pca.page_block_size; // page index within the batch entry
    int32_t const seq_pos_offset = crdM - page_idx_offset*pca.page_block_size; // == crd1 % page_block_size_ -> sequence position within the page
    int32_t const page_idx = pca.block_table[page_idx_offset + crdB*pca.block_table_batch_stride]; // The page index for the given batch and sequence position
    //if (cute::thread0()) {
    //  printf("SM90_TMA_LOAD_MULTICAST_PAGED::copy crdM=%d, crdB=%d, crdK=%d, crdH=%d, page_idx=%d, seq_pos_offset=%d, ptr=%p\n", (int)crdM, (int)crdB, (int) crdK, (int) crdH, (int)page_idx, (int)seq_pos_offset, (void*)desc_ptr);
    //}
    return SM90_TMA_LOAD_MULTICAST_4D::copy(desc_ptr, mbar_ptr, multicast_mask, smem_ptr, crdK, seq_pos_offset, crdH, page_idx);
  }
};



// We also need to specialize Copy_Traits for PAGED_COPY_OP, we can do this by inheriting from the traits of the underlying copy op

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_PAGED_OP : SM90_TMA_LOAD_PAGED {};

// The non-executable SM90_TMA_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD_PAGED, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_PAGED_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask = 0) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {&tma_desc_, &tma_mbar, PagedCopyArgs{} }};
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_PAGED_OP, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc, uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask = 0) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {new_tma_desc, &tma_mbar, PagedCopyArgs{} }};
  }

    CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_PAGED_OP, NumBitsPerTMA>
  with(uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask, PagedCopyArgs const & paged_copy_args ) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {&tma_desc_, &tma_mbar, paged_copy_args }};
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_PAGED_OP, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc, uint64_t& tma_mbar, [[maybe_unused]] uint16_t const& multicast_mask,PagedCopyArgs const &paged_copy_args ) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    return {{}, {new_tma_desc, &tma_mbar, paged_copy_args }};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable SM90_TMA_LOAD with tma_desc and tma_mbar
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_PAGED_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack<SM90_TMA_LOAD_PAGED_OP>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  PagedCopyArgs
  > const opargs_;
};


//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_MULTICAST_PAGED_OP : SM90_TMA_LOAD_MULTICAST_PAGED {};

// The non-executable SM90_TMA_LOAD_MULTICAST with tma_desc and no tma_mbar
// Use .with(tma_mbar, multicast_mask) to construct an executable version
template <class NumBitsPerTMA, class AuxParams_>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED, NumBitsPerTMA, AuxParams_>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  TmaDescriptor tma_desc_;
  using AuxParams = AuxParams_;
  AuxParams aux_params_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr
  TmaDescriptor const*
  get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED_OP, NumBitsPerTMA>
  with(uint64_t& tma_load_mbar, uint16_t const& multicast_mask) const {
    return {{}, {&tma_desc_, &tma_load_mbar, multicast_mask, PagedCopyArgs{} }};
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST_OP with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED_OP, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc, uint64_t& tma_load_mbar, uint16_t const& multicast_mask) const {
    return {{}, {new_tma_desc, &tma_load_mbar, multicast_mask, PagedCopyArgs{} }};
  }

    // Construct an executable SM90_TMA_LOAD_MULTICAST with tma_mbar
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED_OP, NumBitsPerTMA>
  with(uint64_t& tma_load_mbar, uint16_t const& multicast_mask, PagedCopyArgs const& paged_copy_args) const {
    return {{}, {&tma_desc_, &tma_load_mbar, multicast_mask, paged_copy_args }};
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST_OP with tma_mbar (temp. overloaded for grouped gemm/ptr array gemm)
  CUTE_HOST_DEVICE constexpr
  Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED_OP, NumBitsPerTMA>
  with(TmaDescriptor const* new_tma_desc, uint64_t& tma_load_mbar, uint16_t const& multicast_mask, PagedCopyArgs const& paged_copy_args) const {
    return {{}, {new_tma_desc, &tma_load_mbar, multicast_mask, paged_copy_args }};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr
  auto
  get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(aux_params_.g_stride_)>::value);
    return make_counting_tensor(make_layout(g_shape, aux_params_.g_stride_));
  }

  // Don't try to execute a copy with SM90_TMA_LOAD_MULTICAST before calling .with()
  template <class TS, class SLayout,
            class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void
  copy_unpack(Copy_Traits        const& traits,
              Tensor<TS,SLayout> const& src,
              Tensor<TD,DLayout>      & dst) = delete;
};

// The executable SM90_TMA_LOAD_MULTICAST with tma_desc and tma_mbar and multicast_mask
template <class NumBitsPerTMA>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST_PAGED_OP, NumBitsPerTMA>
     : TMA_LOAD_Unpack<SM90_TMA_LOAD_MULTICAST_PAGED_OP>
{
  using ThrID     = Layout<_1>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1,NumBitsPerTMA>>;
  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  tuple<
  TmaDescriptor const*,
  uint64_t*, // smem mbarrier
  uint16_t,   // multicast mask
  PagedCopyArgs,
  > const opargs_;
};


template <class TmaInternalType = void,
          class CopyOp,
          class GEngine, class GLayout,
          class VShape,
          class SLayout,
          class CTA_Tiler,
          class Cluster_Size>
CUTE_HOST_RTC
auto
make_virtualized_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              VShape                  const &virtual_shape,
              SLayout                 const slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
{
    /**
      Variant of cute::make_tma_copy which allows to separate a virtual tensor coordinate space and
      a physical TMA tensor coordinate space. Used for Paged Attention with TMA.
     */
    auto cta_v_tile = make_identity_layout(virtual_shape).compose(cta_tiler);
    auto cta_t_tile = make_layout(cluster_size);
    //cute::print("\nVirtual Shape:"); cute::print(virtual_shape);
    //cute::print("\nPhysical Shape:"); cute::print(gtensor.layout().shape()); cute::print("\n");
    // Prefer TmaInternalType if specified. Fallback to GEngine::value_type
    using TmaType = conditional_t<is_same<void, TmaInternalType>::value, typename GEngine::value_type, TmaInternalType>;
    return detail::make_tma_copy_tiled<TmaType>(copy_op,
                                                gtensor, slayout,
                                                cta_t_tile, cta_v_tile);

}

}
