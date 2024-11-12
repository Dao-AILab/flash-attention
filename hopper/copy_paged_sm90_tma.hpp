#pragma once
#include <cutlass/version.h>

#if CUTLASS_VERSION >= 360
#include "copy_paged_sm90_tma_cutlass36.hpp"
#else 
#include "copy_paged_sm90_tma_cutlass35.hpp"
#endif
