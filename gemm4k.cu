namespace CudaCodeGen {


#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short*>(&(var)))

struct __align__(2) __half {
  __host__ __device__ __half() {}

protected:
  unsigned short __x;
};

__device__ __half __float2half(const float f) {
  __half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
  return val;
}

__device__ float __half2float(const __half h) {
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
  return val;
}


typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int int16_t;
typedef long long int int64_t;

template <typename T, int N>
struct Tensor {
  __device__ T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};

// Specialization for 0-dim case as it does not need size and stride arrays.
// They will be an error as well since zero-length arrays are not allowed.
template <typename T>
struct Tensor<T, 0> {
  __device__ T& operator[](int64_t) {
    return *data;
  };

  T* data;
};


class Philox {
 public:
  __device__ Philox(
      unsigned long long seed,
      unsigned long long subsequence,
      unsigned long long offset) {
    key.x = (unsigned int)seed;
    key.y = (unsigned int)(seed >> 32);
    counter = make_uint4(0, 0, 0, 0);
    counter.z = (unsigned int)(subsequence);
    counter.w = (unsigned int)(subsequence >> 32);
    STATE = 0;
    incr_n(offset / 4);
  }

  __device__ unsigned long operator()() {
    if (STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      for (int i = 0; i < 9; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A);
        key_.y += (kPhilox10B);
      }
      output = single_round(counter_, key_);
      incr();
    }
    unsigned long ret = 0;
    switch (STATE) {
      case 0:
        ret = output.x;
        break;
      case 1:
        ret = output.y;
        break;
      case 2:
        ret = output.z;
        break;
      case 3:
        ret = output.w;
        break;
    }
    STATE = (STATE + 1) % 4;
    return ret;
  }

 private:
  __device__ void incr_n(unsigned long long n) {
    unsigned int nlo = (unsigned int)(n);
    unsigned int nhi = (unsigned int)(n >> 32);
    counter.x += nlo;
    if (counter.x < nlo)
      nhi++;
    counter.y += nhi;
    if (nhi <= counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ void incr() {
    if (++counter.x)
      return;
    if (++counter.y)
      return;
    if (++counter.z)
      return;
    ++counter.w;
  }

  __device__ unsigned int mulhilo32(
      unsigned int a,
      unsigned int b,
      unsigned int* result_high) {
    *result_high = __umulhi(a, b);
    return a * b;
  }

  __device__ uint4 single_round(uint4 ctr, uint2 key) {
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    return ret;
  }

 private:
  static constexpr unsigned long kPhilox10A = 0x9E3779B9;
  static constexpr unsigned long kPhilox10B = 0xBB67AE85;
  static constexpr unsigned long kPhiloxSA = 0xD2511F53;
  static constexpr unsigned long kPhiloxSB = 0xCD9E8D57;

  uint4 counter = {};
  uint4 output = {};
  uint2 key = {};
  unsigned int STATE = 0;
};

__device__ float uniformf(unsigned int x) {
  constexpr float kRanInvM32 = 2.3283064e-10f; // Inverse of 2^32.
  return x * kRanInvM32;
}

__device__ double uniform(unsigned int x, unsigned int y) {
  constexpr double kRan2Pow53Inv = 1.1102230246251565e-16;
  const unsigned long long z =
      (unsigned long long)x ^ ((unsigned long long)y << (53 - 32));
  return z * kRan2Pow53Inv + (kRan2Pow53Inv / 2.0);
}


__device__ constexpr int ceilDiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ constexpr int alignBufferSize(int buffer, int size) {
  return (buffer + (size - 1)) & ~(size - 1);
}

__device__ double clamp(double x, double minv, double maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}

__device__ float clamp(float x, double minv, double maxv) {
  return x < minv ? minv : (x > maxv ? maxv : x);
}

__device__ double frac(double x) {
  return x - trunc(x);
}

__device__ float frac(float x) {
  return x - trunc(x);
}

__device__ double gelu(double x) {
  return x * normcdf(x);
}

__device__ float gelu(float x) {
  return x * normcdf(x);
}

__device__ double reciprocal(double x) {
  return 1 / x;
}

__device__ float reciprocal(float x) {
  return 1 / x;
}

__device__ double relu(double x) {
  return x <= 0 ? 0 : x;
}

__device__ float relu(float x) {
  return x <= 0 ? 0 : x;
}

__device__ double remainder(double a, double b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ float remainder(float a, float b) {
  auto mod = ::fmod(a, b);
  if ((mod != 0) && ((b < 0) != (mod < 0)))
    mod += b;
  return mod;
}

__device__ double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

__device__ float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

__device__ double threshold(double x, double t, double v) {
  return x <= t ? v : x;
}

__device__ float threshold(float x, double t, double v) {
  return x <= t ? v : x;
}

__device__ double where(bool c, double a, double b) {
  return c ? a : b;
}

__device__ float where(bool c, float a, float b) {
  return c ? a : b;
}

__device__ double randLike(Philox rnd) {
  return uniform(rnd(), rnd());
}

__device__ float randLikef(Philox rnd) {
  return uniformf(rnd());
}

// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to 0 it means that dimension doesn't
// participate, otherwise it is the number of threads. We could start with warp
// reductions, then reduce the warps, this could save some shared memory, but
// may actually be slower.
//
//  EXAMPLE USAGE:
//  blockReduceSum<X_THREADS, Y_THREADS, Z_THREADS>
//    (output[output_index], inputs[input_index],
//      [] __device__ (T& a, const T b) { a += b; });
//
// Note: We agressively template functions taking dim3 in the functions below
//       because ROCM uses different types for the various dim3 and maps them
//       directly to intrinsics, but they're dim3 when used after modification.
//
template <
    bool X_REDUCE,
      bool Y_REDUCE,
      bool Z_REDUCE,
      typename T,
      typename Func,
      typename _dim3ti,
      typename _dim3bd>
__device__ void blockReduce(
    T& out,
    const T inp_val,
    Func reduction_op,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    T* shared_mem,
    bool read_write_pred,
    T init_val) {
  unsigned int reduction_size = (X_REDUCE ? block_dim.x : 1) *
      (Y_REDUCE ? block_dim.y : 1) * (Z_REDUCE ? block_dim.z : 1);

  // If this thread will output a final result
  bool should_write = true;

  if (X_REDUCE)
    should_write = should_write && thread_idx.x == 0;
  if (Y_REDUCE)
    should_write = should_write && thread_idx.y == 0;
  if (Z_REDUCE)
    should_write = should_write && thread_idx.z == 0;

  unsigned int reduction_stride;
  unsigned int reduction_tid;
  unsigned int linear_tid;

  if (X_REDUCE && !Y_REDUCE && Z_REDUCE) {
    // Transpose Z and Y in the shared memory so Z and X dims are contiguous in
    // smem
    reduction_stride = 1;
    linear_tid = threadIdx.y * blockDim.z * blockDim.x +
        threadIdx.z * blockDim.x + threadIdx.x;
    reduction_tid = threadIdx.z * blockDim.x + threadIdx.x;
  } else {
    // Normal reduction in order
    reduction_stride =
        (X_REDUCE ? 1
         : (Y_REDUCE ? block_dim.x
            : (Z_REDUCE ? block_dim.x * block_dim.y : 0)));

    linear_tid = thread_idx.z * block_dim.y * block_dim.x +
        thread_idx.y * block_dim.x + thread_idx.x;

    reduction_tid = (Z_REDUCE ? thread_idx.z : 0) *
        (Y_REDUCE ? block_dim.y : 1) * (X_REDUCE ? block_dim.x : 1) +
        (Y_REDUCE ? thread_idx.y : 0) * (X_REDUCE ? block_dim.x : 1) +
        (X_REDUCE ? thread_idx.x : 0);
  }


  if (read_write_pred) {
    shared_mem[linear_tid] = inp_val;
  } else {
    shared_mem[linear_tid] = init_val;
  }
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 = 1 << (31 - __clz(reduction_size));

  if (reduction_tid < np2) {
    if (reduction_tid + np2 < reduction_size) {
      reduction_op(
          shared_mem[linear_tid],
          shared_mem[linear_tid + np2 * reduction_stride]);
    }
  }
  __syncthreads();
  // for (int factor = np2/2; factor > contig_threads / 2; factor>>=1) {
  for (int factor = np2 / 2; factor > 0; factor >>= 1) {
    if (reduction_tid < factor) {
      reduction_op(
          shared_mem[linear_tid],
          shared_mem[linear_tid + factor * reduction_stride]);
    }
    __syncthreads();
  }

  if (should_write && read_write_pred)
    out = shared_mem[linear_tid];
}

// Inter-block reduction.
//
// Function gridReduce performs point-wise reductions of scalars across thread
// blocks. Thread blocks are disjointly partitioned into groups of thread
// blocks, "reduction segments," that are collectively defined by boolean
// template parameters, X_BLOCK, Y_BLOCK and Z_BLOCK. Each of X/Y/Z_BLOCK
// determines whether thread blocks along the dimension should be grouped into
// the same reduction segment. Cross-block reducitons are independently done
// within each segment and generates distinctive results per segment. For
// instance, if all of X/Y/Z_BLOCK are true, reductions will be done across all
// thread blocks since there will be just a single segment consisting of all
// thread blocks. If none of them are true, each thread block will become a
// segment by itself, so no reduction will be performed.
//
// The input scalars to reduce within each segment are a certain subset of
// thread-private scalars provided as part of the gridReduce function
// parameters. Boolean template parameters, X_THREAD, Y_THREAD and Z_THREAD,
// determine which subset of the scalars should be used for inter-block
// reductions. Specifically, all the input scalars of threads along each
// dimension will be used when X/Y/Z_THREAD are true. Otherwise, only the value
// held at offset 0 of each dimension will be used. Thus, for example, if all of
// X/Y/Z_THREAD are true, the scalars of all threads in each block will
// participate in inter-block reductions. If all of them are false, only one
// scalar of the thread at threadIdx.x == threadIdx.y == threadIdx.z == 0 will
// be used. In the code below, we call the subset of threads a "reduction
// block."
//
// Inter-block reductions perform point-wise reductions of scalars of reduction
// blocks within each reduction segment. More specifically, let rb be a
// reduction block and rs be a reduction segment. Let IN(thread_idx, block_idx)
// denote the input scalar of thread at thread_idx and block_idx. The result of
// each reduction segment, OUT(thread_idx, block_idx_out), is defined only for
// each thread_idx in thread block block_idx_out in the segment as follows:
//
//   OUT(thread_idx, block_idx_out) =
//     Reduction of IN(thread_idx, block_idx) for
//       all block_idx in a reduction segment
//
// OUT is not given for all threads that are not in block_idx_out and the
// reduction block.
//
// See also the function comment of gridReduce.

namespace reduction {

// Utility functions
template <typename _dim3>
__device__ __forceinline__ size_t size(const _dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

#define isize(d) d.x* d.y* d.z

template <typename _dim3pos, typename _dim3dim>
__device__ __forceinline__ size_t
offset(const _dim3pos& pos, const _dim3dim& dim) {
  return (size_t)pos.x + (size_t)pos.y * (size_t)dim.x +
      (size_t)pos.z * (size_t)dim.x * (size_t)dim.y;
}

#define ioffset(pos, dim) pos.x + pos.y* dim.x + pos.z* dim.x* dim.y

// Returns dim3 of each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ dim3 dimension_of_reduction_segment(const _dim3& grid_dim) {
  return dim3{X_BLOCK ? grid_dim.x : 1,
    Y_BLOCK ? grid_dim.y : 1,
    Z_BLOCK ? grid_dim.z : 1};
}

// Returns the number of blocks in each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ size_t size_of_reduction_segment(const _dim3& grid_dim) {
  return size(
      dimension_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(grid_dim));
}

// Returns the total number of reduction segments.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__device__ size_t number_of_reduction_segments(const _dim3& grid_dim) {
  return (X_BLOCK ? 1 : grid_dim.x) * (Y_BLOCK ? 1 : grid_dim.y) *
      (Z_BLOCK ? 1 : grid_dim.z);
}

// Returns the 1-D index of the segment of thread block of block_idx.
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      typename _dim3bi,
      typename _dim3gd>
__device__ size_t
index_of_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t seg_idx = 0;
  if (!Z_BLOCK)
    seg_idx += block_idx.z;
  if (!Y_BLOCK)
    seg_idx = seg_idx * grid_dim.y + block_idx.y;
  if (!X_BLOCK)
    seg_idx = seg_idx * grid_dim.x + block_idx.x;
  return seg_idx;
}

// Returns the offset of thread block in its reduction segment.
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      typename _dim3bi,
      typename _dim3gd>
__device__ size_t
offset_in_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t offset = 0;
  if (Z_BLOCK)
    offset = offset * grid_dim.z + block_idx.z;
  if (Y_BLOCK)
    offset = offset * grid_dim.y + block_idx.y;
  if (X_BLOCK)
    offset = offset * grid_dim.x + block_idx.x;
  return offset;
}

// Returns dim3 of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__device__ dim3 dimension_of_reduction_block(const _dim3& block_dim) {
  return dim3{X_THREAD ? block_dim.x : 1,
    Y_THREAD ? block_dim.y : 1,
    Z_THREAD ? block_dim.z : 1};
}

// Returns the number of threads of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__device__ int size_of_reduction_block(const _dim3& block_dim) {
  auto tmp_dim =
      dimension_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(block_dim);
  return isize(tmp_dim);
}

// Returns the linear offset of a thread in a reduction block.
template <
    bool X_THREAD,
      bool Y_THREAD,
      bool Z_THREAD,
      typename _dim3ti,
      typename _dim3bd>
__device__ int offset_in_reduction_block(
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim) {
  int offset = 0;
  if (Z_THREAD)
    offset += thread_idx.z;
  if (Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

// Reduces all the reduction blocks in each reduction segment.
//
// This is only used by one thread block per reduction segment. The input
// reduction blocks of the segment are stored in an intermediate buffer pointed
// by parameter in. Template parameters X/Y/Z_THREAD denote how the reduction
// block is formed.
//
// The size of a reduction block is by definition smaller or equal to the size
// of a thread block. We use the remaining threads to parallelize reductions
// across reduction blocks. For example, when X/Y/Z_THREAD = {true, false,
// false}, we use blockDim.y*blockDim.z threads for each output value. This is
// done first by loading the input values in parallel and then by reducing
// across threads of dimensions whose XYZ_THREAD are false.
//
// Note that what is done here after the loading from global memory is similar
// to what the existing blockReduce function does. The main difference is that
// the logical block to reduce is a 2D domain where the leading dimension is the
// size of a reduction block and the second dimension is the remaining factor in
// each thread block. For example, when X/Y/Z_THREAD = {false, true, false}, the
// threads are arranged as (blockDim.y, blockDim.x*blockDim.z). We do not reduce
// along the first dimension but only the second dimension. So, it is possible
// to reuse the existing blockReduce with dim3{blockDim.y,
// blockDim.x*blockDim.z} instead of blockDim and with X_THREAD and Y_THREAD
// being false and true, respectively. Also, it still need to shuffle the final
// output values to their actual corresponding threads. In the case of when
// X/Y/Z_THREAD = {false, true, false}, after the intra-block reduction, the
// final results will still be held by the first blockDim.y threads, which need
// to be transferred to threads at threadIdx.x == 0 and threadIdx.z == 0.
template <
    bool X_THREAD,
      bool Y_THREAD,
      bool Z_THREAD,
      typename T,
      typename Func>
__device__ void gridReduceLastBlock(
    T& out,
    const T* in,
    const size_t in_size,
    Func reduction_op,
    T* shared_buf,
    bool read_write_pred,
    T init_val) {
  const int tid = ioffset(threadIdx, blockDim);
  const int block_size = isize(blockDim);
  const int rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  T inp = init_val;
  if (tid < in_size) {
    inp = in[tid];
  }
  for (size_t i = tid + block_size; i < in_size; i += block_size) {
    reduction_op(inp, in[i]);
  }

  const auto should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);

  auto rem_size = block_size / rblock_size;

  if (rem_size > 1) {
    const int rblock_offset = tid % rblock_size;
    const int rblock_idx = tid / rblock_size;
    blockReduce<false, true, false>(
        inp,
        inp,
        reduction_op,
        dim3{(unsigned)rblock_offset, (unsigned)rblock_idx, 0},
        dim3{(unsigned)rblock_size, (unsigned)rem_size},
        shared_buf,
        true,
        init_val);
    __syncthreads();
    if (tid < rblock_size) {
      shared_buf[tid] = inp;
    }
    __syncthreads();
    if (should_write) {
      inp = shared_buf[offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
          threadIdx, blockDim)];
    }
  }

  if (should_write && read_write_pred) {
    out = inp;
  }
}

// Reduces per-thread values across thread blocks.
//
// Function parameters:
// - out: Per-thread output location
// - inp_val: Per-thread input value
// - reduction_op: Scalar reduction function
// - work_buf: Temporary buffer for cross-block reductions
// - sync_flags: A vector of integers for synchronizations
// - shared_buf: Shared memory buffer for intra-block reduction
//
// Return true when the thread block has the valid result.
//
// Template parameters:
// - X/Y/Z_BLOCK: When true, reduces across thread blocks along the X/Y/Z
//   dimensions
// - X/Y/Z_THREAD: When true, all threads along the X/Y/Z dimensions participate
//   in the cross-block reduction. Otherwise, only threads at offset 0 do.
// - T: Scalar data type of input/output data
// - Func: Type of scalara reduction function
//
// Template parameters X/Y/Z_BLOCK define a group of thread blocks that are
// reduced together. We call it a reduction segment. Some examples are:
//
// Case 1: X/Y/Z_BLOCK == true/true/true -> There is only one segment, which
// includes all thread blocks. It is effecively the same as the grid.
//
// Case 2: X/Y/Z_BLOCK == false/false/false -> Each thread block comprises an
// individual segment by itself.
//
// Case 3: X/Y/Z_BLOCK == true/false/false -> Each segment contains thread
// blocks that have the same blockDim.x. There will be blockDim.y*blockDim.z
// such segments.
//
// X/Y/Z_THREAD defines a sub region of a thread block that should be reduced
// with the sub regions of other thread blocks. We call it a reduction block.
// E.g.,
//
// Case 1: X/Y/Z_THREAD == false/false/false -> Only thread 0 participates in
// the cross-block reductions. The reduction block is 1x1x1 with thread 0.
//
// Case 2: X/Y/Z_THREAD == true/true/true-> All threads in a thread block
// participate in the cross-block reductions. The reduction block in this case
// is equivalent to the thread block.
//
// After the function completes, only one thread block per reduction segment
// gets valid reduction results. There is no guarantee which particular block
// gets the final results.
//
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      bool X_THREAD,
      bool Y_THREAD,
      bool Z_THREAD,
      typename T,
      typename Func>
__device__ bool gridReduce(
    T& out,
    T inp_val,
    Func reduction_op,
    volatile T* work_buf,
    Tensor<int64_t, 1> sync_flags,
    T* shared_buf,
    bool read_write_pred,
    T init_val) {
  // Number of values to reduce in the grid dimensions
  const auto seg_size =
      size_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the seg_size
  const auto seg_idx =
      index_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  work_buf += seg_idx * seg_size * rblock_size;

  if ((X_THREAD || threadIdx.x == 0) && (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto rblock_offset = offset_in_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(
        blockIdx, gridDim);
    auto thread_offset =
        offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset = rblock_size * rblock_offset + thread_offset;
    if (read_write_pred) {
      work_buf[work_buf_offset] = inp_val;
    } else {
      work_buf[work_buf_offset] = init_val;
    }
  }
  __syncthreads();

  __shared__ bool last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    // printf("%ld\n", sync_flags[seg_idx]);
    auto old = (int64_t)atomicAdd((unsigned long long*)&sync_flags[seg_idx], 1);
    last_block = old + 1 == seg_size;
    // printf("Last_block = %d + 1 == %d\n", (int)old, (int)seg_size);
  }
  __syncthreads();

  if (last_block) {
    // printf("Last block %d %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    // final reduction
    gridReduceLastBlock<X_THREAD, Y_THREAD, Z_THREAD>(
        out,
        (T*)work_buf,
        seg_size * rblock_size,
        reduction_op,
        shared_buf,
        read_write_pred,
        init_val);
    return true;
  } else {
    // printf("Not last block %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    return false;
  }
}

} // namespace reduction


namespace broadcast {

template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD>
__host__ __device__ unsigned offset_of_source(
    const dim3& block_dim,
    const dim3& thread_idx) {
  unsigned offset = 0;
  if (!Z_THREAD)
    offset = offset * block_dim.z + thread_idx.z;
  if (!Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (!X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

// Broadcasts within partitioned groups of threads.
//
// X_THREAD: Broadcast from threadIdx.x == 0 if true
// Y_THREAD: Broadcast from threadIdx.y == 0 if true
// Z_THREAD: Broadcast from threadIdx.z == 0 if true
// inp_val: Per-thread source value. Only valid when the thread is a source.
// out: Per-thread output location
//
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T>
__device__ void blockBroadcast(T& out, T inp_val, T* shared_mem) {
  const bool has_valid_data = (!X_THREAD || threadIdx.x == 0) &&
      (!Y_THREAD || threadIdx.y == 0) && (!Z_THREAD || threadIdx.z == 0);

  const auto shared_offset =
      offset_of_source<X_THREAD, Y_THREAD, Z_THREAD>(blockDim, threadIdx);

  if (has_valid_data)
    shared_mem[shared_offset] = inp_val;

  __syncthreads();

  out = shared_mem[shared_offset];
}

} // namespace broadcast

// -----------------------------------------------------------------------------------------------
//  Block Welford Primitives
// -----------------------------------------------------------------------------------------------
// Basic utility for welford update. Can be used to scan one value, or two merge
// two welford results
template <typename T, typename TN>
__inline__ __device__ void welfordCombine(
    T& a_M2,
    T& a_avg,
    TN& a_N,
    const T& b_M2,
    const T& b_avg,
    TN b_N) {
  TN ab_N = a_N + b_N;
  T delta = b_avg - a_avg;
  a_avg += delta * b_N / ab_N;
  a_M2 += b_M2 + delta * delta * a_N * b_N / ab_N;
  a_N = ab_N;
}

// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block.
template <
    bool X_REDUCE,
      bool Y_REDUCE,
      bool Z_REDUCE,
      typename T,
      typename TN,
      typename _dim3ti,
      typename _dim3bd>
__inline__ __device__ void blockWelford(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    const T& in_M2,
    const T& in_avg,
    const TN in_N,
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim,
    T* shared_mem_M2,
    T* shared_mem_avg,
    TN* shared_mem_N,
    bool read_write_pred,
    T init_val) {
  unsigned int reduction_size = (X_REDUCE ? block_dim.x : 1) *
      (Y_REDUCE ? block_dim.y : 1) * (Z_REDUCE ? block_dim.z : 1);
  // If this thread will output a final result
  bool should_write = true;
  if (X_REDUCE)
    should_write = should_write && thread_idx.x == 0;
  if (Y_REDUCE)
    should_write = should_write && thread_idx.y == 0;
  if (Z_REDUCE)
    should_write = should_write && thread_idx.z == 0;
  unsigned int reduction_stride;
  unsigned int reduction_tid;
  unsigned int linear_tid;
  if (X_REDUCE && !Y_REDUCE && Z_REDUCE) {
    // Transpose Z and Y in the shared memory so Z and X dims are contiguous in
    // smem
    reduction_stride = 1;
    linear_tid = threadIdx.y * blockDim.z * blockDim.x +
        threadIdx.z * blockDim.x + threadIdx.x;
    reduction_tid = threadIdx.z * blockDim.x + threadIdx.x;
  } else {
    // Normal reduction in order
    reduction_stride =
        (X_REDUCE ? 1
         : (Y_REDUCE ? block_dim.x
            : (Z_REDUCE ? block_dim.x * block_dim.y : 0)));
    linear_tid = thread_idx.z * block_dim.y * block_dim.x +
        thread_idx.y * block_dim.x + thread_idx.x;
    reduction_tid = (Z_REDUCE ? thread_idx.z : 0) *
        (Y_REDUCE ? block_dim.y : 1) * (X_REDUCE ? block_dim.x : 1) +
        (Y_REDUCE ? thread_idx.y : 0) * (X_REDUCE ? block_dim.x : 1) +
        (X_REDUCE ? thread_idx.x : 0);
  }
  if (read_write_pred) {
    shared_mem_M2[linear_tid] = in_M2;
    shared_mem_avg[linear_tid] = in_avg;
    shared_mem_N[linear_tid] = in_N;
  } else {
    shared_mem_M2[linear_tid] = init_val;
    shared_mem_avg[linear_tid] = init_val;
    shared_mem_N[linear_tid] = 0;
  }
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 = 1 << (31 - __clz(reduction_size));
  if (reduction_tid < np2) {
    if (reduction_tid + np2 < reduction_size) {
      welfordCombine(
          shared_mem_M2[linear_tid],
          shared_mem_avg[linear_tid],
          shared_mem_N[linear_tid],
          shared_mem_M2[linear_tid + np2 * reduction_stride],
          shared_mem_avg[linear_tid + np2 * reduction_stride],
          shared_mem_N[linear_tid + np2 * reduction_stride]);
    }
  }
  __syncthreads();
  for (int factor = np2 / 2; factor > 0; factor >>= 1) {
    if (reduction_tid < factor) {
      welfordCombine(
          shared_mem_M2[linear_tid],
          shared_mem_avg[linear_tid],
          shared_mem_N[linear_tid],
          shared_mem_M2[linear_tid + factor * reduction_stride],
          shared_mem_avg[linear_tid + factor * reduction_stride],
          shared_mem_N[linear_tid + factor * reduction_stride]);
    }
    __syncthreads();
  }
  if (should_write && read_write_pred) {
    out_M2 = shared_mem_M2[linear_tid];
    out_avg = shared_mem_avg[linear_tid];
    out_N = shared_mem_N[linear_tid];
  }
}
// -----------------------------------------------------------------------------------------------
//  Grid Welford Prototype
// -----------------------------------------------------------------------------------------------
namespace welford {
// Utility functions
template <typename _dim3>
__host__ __device__ __forceinline__ size_t size(const _dim3& d) {
  return (size_t)d.x * (size_t)d.y * (size_t)d.z;
}

#define isize(d) d.x* d.y* d.z

template <typename _dim3pos, typename _dim3dim>
__host__ __device__ __forceinline__ size_t
offset(const _dim3pos& pos, const _dim3dim& dim) {
  return (size_t)pos.x + (size_t)pos.y * (size_t)dim.x +
      (size_t)pos.z * (size_t)dim.x * (size_t)dim.y;
}

#define ioffset(pos, dim) pos.x + pos.y* dim.x + pos.z* dim.x* dim.y

// Returns dim3 of each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ dim3 dimension_of_reduction_segment(const _dim3& grid_dim) {
  return dim3{X_BLOCK ? grid_dim.x : 1,
    Y_BLOCK ? grid_dim.y : 1,
    Z_BLOCK ? grid_dim.z : 1};
}

// Returns the number of blocks in each reduction segment.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ size_t size_of_reduction_segment(const _dim3& grid_dim) {
  return size(
      dimension_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(grid_dim));
}

// Returns the total number of reduction segments.
template <bool X_BLOCK, bool Y_BLOCK, bool Z_BLOCK, typename _dim3>
__host__ __device__ size_t number_of_reduction_segments(const _dim3& grid_dim) {
  return (X_BLOCK ? 1 : grid_dim.x) * (Y_BLOCK ? 1 : grid_dim.y) *
      (Z_BLOCK ? 1 : grid_dim.z);
}

// Returns the 1-D index of the segment of thread block of block_idx.
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      typename _dim3bi,
      typename _dim3gd>
__host__ __device__ size_t
index_of_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t seg_idx = 0;
  if (!Z_BLOCK)
    seg_idx += block_idx.z;
  if (!Y_BLOCK)
    seg_idx = seg_idx * grid_dim.y + block_idx.y;
  if (!X_BLOCK)
    seg_idx = seg_idx * grid_dim.x + block_idx.x;
  return seg_idx;
}

// Returns the offset of thread block in its reduction segment.
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      typename _dim3bi,
      typename _dim3gd>
__host__ __device__ size_t
offset_in_reduction_segment(const _dim3bi& block_idx, const _dim3gd& grid_dim) {
  size_t offset = 0;
  if (Z_BLOCK)
    offset = offset * grid_dim.z + block_idx.z;
  if (Y_BLOCK)
    offset = offset * grid_dim.y + block_idx.y;
  if (X_BLOCK)
    offset = offset * grid_dim.x + block_idx.x;
  return offset;
}

// Returns dim3 of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__host__ __device__ dim3 dimension_of_reduction_block(const _dim3& block_dim) {
  return dim3{X_THREAD ? block_dim.x : 1,
    Y_THREAD ? block_dim.y : 1,
    Z_THREAD ? block_dim.z : 1};
}

// Returns the number of threads of each reduction block.
template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename _dim3>
__host__ __device__ int size_of_reduction_block(const _dim3& block_dim) {
  auto tmp_dim =
      dimension_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(block_dim);
  return isize(tmp_dim);
}

// Returns the linear offset of a thread in a reduction block.
template <
    bool X_THREAD,
      bool Y_THREAD,
      bool Z_THREAD,
      typename _dim3ti,
      typename _dim3bd>
__host__ __device__ int offset_in_reduction_block(
    const _dim3ti& thread_idx,
    const _dim3bd& block_dim) {
  int offset = 0;
  if (Z_THREAD)
    offset += thread_idx.z;
  if (Y_THREAD)
    offset = offset * block_dim.y + thread_idx.y;
  if (X_THREAD)
    offset = offset * block_dim.x + thread_idx.x;
  return offset;
}

template <bool X_THREAD, bool Y_THREAD, bool Z_THREAD, typename T, typename TN>
__device__ void gridWelfordLastBlock(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    const T* in_M2,
    const T* in_avg,
    const TN* in_N,
    const size_t in_size,
    T* shared_buf_M2,
    T* shared_buf_avg,
    TN* shared_buf_N,
    bool read_write_pred,
    T init_val) {
  const int tid = ioffset(threadIdx, blockDim);
  const int block_size = isize(blockDim);
  const int rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  T inp_M2 = init_val;
  T inp_avg = init_val;
  TN inp_N = 0;
  if (tid < in_size) {
    inp_M2 = in_M2[tid];
    inp_avg = in_avg[tid];
    inp_N = in_N[tid];
  }
  for (size_t i = tid + block_size; i < in_size; i += block_size) {
    welfordCombine(inp_M2, inp_avg, inp_N, in_M2[i], in_avg[i], in_N[i]);
  }
  const auto should_write = (X_THREAD || threadIdx.x == 0) &&
      (Y_THREAD || threadIdx.y == 0) && (Z_THREAD || threadIdx.z == 0);

  auto rem_size = block_size / rblock_size;

  if (rem_size > 1) {
    const int rblock_offset = tid % rblock_size;
    const int rblock_idx = tid / rblock_size;
    blockWelford<false, true, false>(
        inp_M2,
        inp_avg,
        inp_N,
        inp_M2,
        inp_avg,
        inp_N,
        dim3{(unsigned)rblock_offset, (unsigned)rblock_idx, 0},
        dim3{(unsigned)rblock_size, (unsigned)rem_size},
        shared_buf_M2,
        shared_buf_avg,
        shared_buf_N,
        true,
        init_val);
    __syncthreads();
    if (tid < rblock_size) {
      shared_buf_M2[tid] = inp_M2;
      shared_buf_avg[tid] = inp_avg;
      shared_buf_N[tid] = inp_N;
    }
    __syncthreads();
    if (should_write) {
      size_t offset_write =
          offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
              threadIdx, blockDim);
      inp_M2 = shared_buf_M2[offset_write];
      inp_avg = shared_buf_avg[offset_write];
      inp_N = shared_buf_N[offset_write];
    }
  }

  if (should_write && read_write_pred) {
    out_M2 = inp_M2;
    out_avg = inp_avg;
    out_N = inp_N;
  }
}

//    Grid welford combine
template <
    bool X_BLOCK,
      bool Y_BLOCK,
      bool Z_BLOCK,
      bool X_THREAD,
      bool Y_THREAD,
      bool Z_THREAD,
      typename T,
      typename TN>
__device__ bool gridWelford(
    T& out_M2,
    T& out_avg,
    TN& out_N,
    T inp_M2,
    T inp_avg,
    TN inp_N,
    volatile T* work_buf_M2,
    volatile T* work_buf_avg,
    volatile TN* work_buf_N,
    Tensor<int64_t, 1> sync_flags,
    T* shared_buf_M2,
    T* shared_buf_avg,
    TN* shared_buf_N,
    bool read_write_pred,
    T init_val) {
  // Number of values to reduce in the grid dimensions
  const auto seg_size =
      size_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(gridDim);

  // Index of the reduction we're performing out of the seg_size
  const auto seg_idx =
      index_of_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(blockIdx, gridDim);

  // Number of threads we can use in final reduction, Seems to assume all
  // threads in the block participate
  const auto rblock_size =
      size_of_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(blockDim);

  // advance to the offset for this segment
  // index of reduction * size of the reduction * size of threads
  shared_buf_M2 += seg_idx * seg_size * rblock_size;
  shared_buf_avg += seg_idx * seg_size * rblock_size;
  shared_buf_N += seg_idx * seg_size * rblock_size;
  if ((X_THREAD || threadIdx.x == 0) && (Y_THREAD || threadIdx.y == 0) &&
      (Z_THREAD || threadIdx.z == 0)) {
    auto rblock_offset = offset_in_reduction_segment<X_BLOCK, Y_BLOCK, Z_BLOCK>(
        blockIdx, gridDim);
    auto thread_offset =
        offset_in_reduction_block<X_THREAD, Y_THREAD, Z_THREAD>(
            threadIdx, blockDim);
    auto work_buf_offset = rblock_size * rblock_offset + thread_offset;
    if (read_write_pred) {
      work_buf_M2[work_buf_offset] = inp_M2;
      work_buf_avg[work_buf_offset] = inp_avg;
      work_buf_N[work_buf_offset] = inp_N;
    } else {
      work_buf_M2[work_buf_offset] = init_val;
      work_buf_avg[work_buf_offset] = init_val;
      work_buf_N[work_buf_offset] = 0;
    }
  }
  __syncthreads();

  __shared__ bool last_block;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    __threadfence();
    auto old = (int64_t)atomicAdd((unsigned long long*)&sync_flags[seg_idx], 1);
    last_block = old + 1 == seg_size;
  }
  __syncthreads();

  if (last_block) {
    // final reduction
    gridWelfordLastBlock<X_THREAD, Y_THREAD, Z_THREAD>(
        out_M2,
        out_avg,
        out_N,
        (T*)work_buf_M2,
        (T*)work_buf_avg,
        (TN*)work_buf_N,
        seg_size * rblock_size,
        shared_buf_M2,
        shared_buf_avg,
        shared_buf_N,
        read_write_pred,
        init_val);
    return true;
  } else {
    return false;
  }
}
} // namespace welford
__global__ void kernel1(Tensor<float, 2> T0, Tensor<float, 2> T1, Tensor<float, 2> T5) {
  float T8[((ceilDiv(128, 16)) * (ceilDiv(128, 16)))] = {0};
  for(size_t ki49 = 0; ki49 < (ceilDiv(4096, 8)); ++ki49) {
    __shared__ float T2[((ceilDiv((128 * 8), 256)) * 256)];
    __shared__ float T3[((ceilDiv((8 * 128), 256)) * 256)];
    for(size_t ki55 = 0; ki55 < (ceilDiv((128 * 8), 256)); ++ki55) {
      T2[((((ki55 * 256) + threadIdx.x) / 8) * 8) + (((ki55 * 256) + threadIdx.x) % 8)]
          = T0[(((blockIdx.x * 128) + (((ki55 * 256) + threadIdx.x) / 8)) * T0.stride[0]) + (((ki49 * 8) + (((ki55 * 256) + threadIdx.x) % 8)) * T0.stride[1])];
    }
    for(size_t ki83 = 0; ki83 < (ceilDiv((8 * 128), 256)); ++ki83) {
      T3[((((ki83 * 256) + threadIdx.x) / 128) * 128) + (((ki83 * 256) + threadIdx.x) % 128)]
          = T1[(((ki49 * 8) + (((ki83 * 256) + threadIdx.x) / 128)) * T1.stride[0]) + (((blockIdx.y * 128) + (((ki83 * 256) + threadIdx.x) % 128)) * T1.stride[1])];
    }
    __syncthreads();
    for(size_t ki108 = 0; ki108 < 8; ++ki108) {
      float T6[(ceilDiv(128, 16))];
      float T7[(ceilDiv(128, 16))];
      for(size_t ki119 = 0; ki119 < (ceilDiv(128, 16)); ++ki119) {
        T6[ki119]
            = T2[(((ki119 * 16) + (threadIdx.x / 16)) * 8) + ki108];
      }
      for(size_t ki154 = 0; ki154 < (ceilDiv(128, 16)); ++ki154) {
        T7[ki154]
            = T3[(ki108 * ((ceilDiv(128, 16)) * 16)) + ((ki154 * 16) + (threadIdx.x % 16))];
      }
      if ((((((ki49 * 8) + ki108) < 4096) && (((blockIdx.y * ((ceilDiv(128, 16)) * 16)) + ((((ceilDiv(128, 16)) - 1) * 16) + (threadIdx.x % 16))) < 4096)) && (((blockIdx.x * ((ceilDiv(128, 16)) * 16)) + ((((ceilDiv(128, 16)) - 1) * 16) + (threadIdx.x / 16))) < 4096))) {
        for(size_t ki178 = 0; ki178 < (ceilDiv(128, 16)); ++ki178) {
          for(size_t ki182 = 0; ki182 < (ceilDiv(128, 16)); ++ki182) {
            float T4[1];
            T4[0]
                = T6[ki178]
                * T7[ki182];
            T8[(ki178 * (ceilDiv(128, 16))) + ki182]
                = T8[(ki178 * (ceilDiv(128, 16))) + ki182]
                + T4[0];
          }
        }
      } else {
        for(size_t ki178 = 0; ki178 < (ceilDiv(128, 16)); ++ki178) {
          for(size_t ki182 = 0; ki182 < (ceilDiv(128, 16)); ++ki182) {
            float T4[1];
            if ((((((blockIdx.x * ((ceilDiv(128, 16)) * 16)) + ((ki178 * 16) + (threadIdx.x / 16))) < 4096) && (((ki49 * 8) + ki108) < 4096)) && (((blockIdx.y * ((ceilDiv(128, 16)) * 16)) + ((ki182 * 16) + (threadIdx.x % 16))) < 4096))) {
              T4[0]
                  = T6[ki178]
                  * T7[ki182];
            }
            if ((((((blockIdx.x * ((ceilDiv(128, 16)) * 16)) + ((ki178 * 16) + (threadIdx.x / 16))) < 4096) && (((ki49 * 8) + ki108) < 4096)) && (((blockIdx.y * ((ceilDiv(128, 16)) * 16)) + ((ki182 * 16) + (threadIdx.x % 16))) < 4096))) {
              T8[(ki178 * (ceilDiv(128, 16))) + ki182]
                  = T8[(ki178 * (ceilDiv(128, 16))) + ki182]
                  + T4[0];
            }
          }
        }
      }
    }
    __syncthreads();
  }
  for(size_t ki229 = 0; ki229 < (ceilDiv(128, 16)); ++ki229) {
    __shared__ float T9[((16 * 16) * (ceilDiv(128, 16)))];
    for(size_t ki239 = 0; ki239 < (ceilDiv(128, 16)); ++ki239) {
      if (((((blockIdx.x * ((ceilDiv(128, 16)) * 16)) + ((ki229 * 16) + (threadIdx.x / 16))) < 4096) && (((blockIdx.y * ((ceilDiv(128, 16)) * 16)) + ((ki239 * 16) + (threadIdx.x % 16))) < 4096))) {
        T9[((threadIdx.x / 16) * ((ceilDiv(128, 16)) * 16)) + ((ki239 * 16) + (threadIdx.x % 16))]
            = T8[(ki229 * (ceilDiv(128, 16))) + ki239];
      }
    }
    __syncthreads();
    for(size_t ki253 = 0; ki253 < (ceilDiv(((16 * 16) * (ceilDiv(128, 16))), 256)); ++ki253) {
      if (((((blockIdx.x * 128) + ((ki229 * 16) + ((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) / 16))) < 4096) && (((blockIdx.y * 128) + (((((ki253 * 256) + threadIdx.x) % (ceilDiv(128, 16))) * 16) + ((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) % 16))) < 4096))) {
        T5[(((blockIdx.x * 128) + ((ki229 * 16) + ((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) / 16))) * T5.stride[0]) + ((blockIdx.y * 128) + (((((ki253 * 256) + threadIdx.x) % (ceilDiv(128, 16))) * 16) + ((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) % 16)))]
            = T9[(((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) / 16) * ((ceilDiv(128, 16)) * 16)) + (((((ki253 * 256) + threadIdx.x) % (ceilDiv(128, 16))) * 16) + ((((ki253 * 256) + threadIdx.x) / (ceilDiv(128, 16))) % 16))];
      }
    }
    __syncthreads();
  }
}
}
