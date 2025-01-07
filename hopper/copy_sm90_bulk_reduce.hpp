/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include<cute/arch/copy_sm90_tma.hpp>

namespace cute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM90_BULK_REDUCE_ADD
{
  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }

  CUTE_HOST_DEVICE static void
  copy(float const* smem_ptr,
       float      * gmem_ptr, int32_t store_bytes, uint64_t cache_hint)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.add.f32 [%0], [%1], %2, %3;\n"
                     :
                     : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes), "l"(cache_hint)
                     : "memory");
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use BULK_REDUCE_ADD without CUTE_ARCH_TMA_SM90_ENABLED.");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
