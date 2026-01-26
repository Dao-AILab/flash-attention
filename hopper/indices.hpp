#pragma once

#include <cstdint>

namespace flash {

class Range {
public:
  using value_type = int;

  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Range::value_type;
    using reference = Range::value_type;  // not lvalue

    __forceinline__ __device__ iterator(int cur, int end) : cur_(cur), end_(end) {}
    __forceinline__ __device__ reference operator*() { return cur_; }
    __forceinline__ __device__ iterator& operator++() { ++cur_; return *this; }

    __forceinline__ __device__ bool operator==(iterator const& other) const { return cur_ == other.cur_ && end_ == other.end_; }
    __forceinline__ __device__ bool operator!=(iterator const& other) const { return !operator==(other); }

  private:
    int cur_;
    int end_;
  };

  class reverse_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Range::value_type;
    using reference = Range::value_type;  // not lvalue

    __forceinline__ __device__ reverse_iterator(int cur, int end) : cur_(cur), end_(end) {}
    __forceinline__ __device__ reference operator*() { return cur_; }
    __forceinline__ __device__ reverse_iterator& operator++() { --cur_; return *this; }

    __forceinline__ __device__ bool operator==(reverse_iterator const& other) const { return cur_ == other.cur_ && end_ == other.end_; }
    __forceinline__ __device__ bool operator!=(reverse_iterator const& other) const { return !operator==(other); }

  private:
    int cur_;
    int end_;
  };

  __forceinline__ __device__ Range(int start, int end) : start_{start}, end_{end} {}
  __forceinline__ __device__ iterator begin() const { return iterator(0, end_); }
  __forceinline__ __device__ iterator end()   const { return iterator(end_, end_); }

  __forceinline__ __device__ reverse_iterator rbegin() const { return reverse_iterator(end_ - 1, start_ - 1); }
  __forceinline__ __device__ reverse_iterator rend()   const { return reverse_iterator(start_ - 1, start_ - 1); }

private:
  int start_;
  int end_;
};


namespace detail {

__forceinline__ __device__ uint32_t bits_reverse(uint32_t bits) {
    asm volatile("brev.b32 %0, %0;" : "+r"(bits):);
    return bits;
}

}  // namespace detail

/// Sparse indices from Compressed Row Mask
template <typename MaskType>
class SparseIndicesCRM {
public:
  using value_type = int;
  using mask_type = MaskType;

  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = SparseIndicesCRM::value_type;
    using reference = SparseIndicesCRM::value_type;  // not lvalue

    __forceinline__ __device__ iterator(mask_type const* ptr, mask_type const* end)
      : offset_(-sizeof(mask_type) * 8), val_(0), ptr_(ptr), end_(end) {
      maybe_advance();
    }
    __forceinline__ __device__ reference operator*() {
      if (val_ == 0) {
        return std::numeric_limits<reference>::max();
      }
      auto pos = __ffs(val_) - 1;
      return offset_ + pos;
    }
    __forceinline__ __device__ iterator& operator++() {
      val_ &= val_ - 1;
      maybe_advance();
      return *this;
    }

    __forceinline__ __device__ bool operator==(iterator const& other) const { return val_ == other.val_ && ptr_ == other.ptr_; }
    __forceinline__ __device__ bool operator!=(iterator const& other) const { return !operator==(other); }

  // private:
    __forceinline__ __device__ void maybe_advance() {
      while (val_ == 0 && ptr_ != end_) {
        val_ = *ptr_++;
        offset_ += sizeof(mask_type) * 8;
      }
    }

    int offset_;
    uint32_t val_;
    mask_type const* ptr_;
    mask_type const* end_;
  };

  class reverse_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = SparseIndicesCRM::value_type;
    using reference = SparseIndicesCRM::value_type;  // not lvalue

    __forceinline__ __device__ reverse_iterator(mask_type const* ptr, mask_type const* end)
        : offset_((ptr - end) * sizeof(mask_type) * 8), val_(0), ptr_(ptr), end_(end) {
      maybe_advance();
    }
    __forceinline__ __device__ reference operator*() {
      if (val_ == 0) {
        return std::numeric_limits<reference>::min();
      }
      auto pos = __ffs(val_) - 1;
      return offset_ + (31 - pos);
    }
    __forceinline__ __device__ reverse_iterator& operator++() {
      val_ &= val_ - 1;
      maybe_advance();
      return *this;
    }

    __forceinline__ __device__ bool operator==(reverse_iterator const& other) const { return val_ == other.val_ && ptr_ == other.ptr_; }
    __forceinline__ __device__ bool operator!=(reverse_iterator const& other) const { return !operator==(other); }

  // private:
    __forceinline__ __device__ void maybe_advance() {
      while (val_ == 0 && ptr_ != end_) {
        val_ = detail::bits_reverse(*ptr_--);
        offset_ -= sizeof(mask_type) * 8;
      }
    }

    int offset_;
    uint32_t val_;
    mask_type const* ptr_;
    mask_type const* end_;
  };

  __forceinline__ __device__ SparseIndicesCRM(mask_type const* begin, mask_type const* end) : begin_{begin}, end_{end} {}
  __forceinline__ __device__ iterator begin() const { return iterator(begin_, end_); }
  __forceinline__ __device__ iterator end()   const { return iterator(end_,   end_); }

  __forceinline__ __device__ reverse_iterator rbegin() const { return reverse_iterator(end_  - 1,  begin_ - 1); }
  __forceinline__ __device__ reverse_iterator rend()   const { return reverse_iterator(begin_- 1,  begin_ - 1); }

private:
  mask_type const* begin_;
  mask_type const* end_;
};

}
