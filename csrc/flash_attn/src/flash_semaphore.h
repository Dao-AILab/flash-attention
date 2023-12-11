// modify from: cutass/include/cutlass/semaphore.h

#include "cuda_runtime.h"

namespace flash {

/// CTA-wide semaphore for inter-CTA synchronization.
class FlashSemaphore { 
public:

  bool wait_thread;
  int state;

public:

  /// Implements a semaphore to wait for a flag to reach a given value
  inline __device__ FlashSemaphore(int thread_id): 
    wait_thread(thread_id < 0 || thread_id == 0),
    state(-1) {

  }

  /// Permit fetching the synchronization mechanism early
  inline __device__ void fetch(int* lock) {
    if (wait_thread) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #else
      asm volatile ("ld.global.cg.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));  
      #endif
    }
  }

  /// Gets the internal state
  inline __device__ int get_state() const {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  inline __device__ void wait(int* lock, int status = 0) {
    while( __syncthreads_and(state != status) ) {
        fetch(lock);
    }
    __syncthreads();
  }

  /// Updates the lock with the given result
  inline __device__ void release(int* lock, int status = 0) {
    __syncthreads();

    if (wait_thread) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
      asm volatile ("st.global.release.gpu.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #else
      asm volatile ("st.global.cg.b32 [%0], %1;\n" : : "l"(lock), "r"(status));
      #endif
    }
  }
};

} // namespace flash