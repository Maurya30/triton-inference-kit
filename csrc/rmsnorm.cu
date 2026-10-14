// Fused RMSNorm, CUDA C++ port of src/triton_inference_kit/rmsnorm.py.
//
//     y[r, i] = w[i] * x[r, i] / sqrt( mean_j(x[r, j]^2) + eps )
//
// Design (docs/cuda_rmsnorm.md walks through every piece of this):
//
//   * One thread block per row, grid = (M,). Rows are independent, so the
//     only reduction is inside one block.
//   * Pass 1: each thread accumulates a partial sum of squares in fp32 over
//     a strided slice of the row. The block then reduces: warp shuffle with
//     __shfl_down_sync -> one float per warp in shared memory -> warp 0
//     shuffles those partials -> thread 0 owns the row total.
//   * Thread 0 computes rstd = rsqrtf(total / N + eps) and broadcasts it
//     through shared memory.
//   * Pass 2: each thread re-reads its slice of x, multiplies by rstd and by
//     the weight, converts back to the storage dtype and stores.
//   * Memory access is 16 bytes per thread per instruction (one float4) when
//     the row, the weight and the output are 16-byte aligned. The N % VEC
//     leftover elements go through a scalar tail. Rows that are not 16-byte
//     aligned (N * sizeof(T) not a multiple of 16, or a tensor view with a
//     storage offset) take an all-scalar path. The choice is made per block,
//     so it is uniform across the block and causes no warp divergence.
//
// Supported dtypes: float16, bfloat16, float32 (same dtype in and out, and
// for the weight). Accumulation is always fp32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>

namespace {

constexpr int kWarpSize = 32;
constexpr int kMinThreadsPerBlock = 128;   // 4 warps: enough to hide latency on small rows
constexpr int kMaxThreadsPerBlock = 1024;  // hardware limit; also keeps num_warps <= 32
constexpr int kVecBytes = 16;              // one float4 = one 128-bit load/store

// ---------------------------------------------------------------------------
// Per-dtype helpers: how many elements fit in 16 bytes, and fp32 conversion.
// ---------------------------------------------------------------------------
template <typename T>
struct DType;

template <>
struct DType<float> {
  static constexpr int kVec = kVecBytes / sizeof(float);  // 4
  __device__ static __forceinline__ float to_float(float v) { return v; }
  __device__ static __forceinline__ float from_float(float v) { return v; }
};

template <>
struct DType<__half> {
  static constexpr int kVec = kVecBytes / sizeof(__half);  // 8
  __device__ static __forceinline__ float to_float(__half v) { return __half2float(v); }
  __device__ static __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
};

template <>
struct DType<__nv_bfloat16> {
  static constexpr int kVec = kVecBytes / sizeof(__nv_bfloat16);  // 8
  __device__ static __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  __device__ static __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }
};

// 16 bytes viewed either as one float4 (the load/store instruction operates
// on this member) or as VEC scalars (the arithmetic operates on these).
// Type punning through a union is the idiomatic CUDA way to unpack a
// vector load; __half and __nv_bfloat16 have trivial default constructors
// in C++11 and later, so they are allowed as union members.
template <typename T>
union alignas(kVecBytes) Pack {
  float4 raw;
  T elems[DType<T>::kVec];
};
static_assert(sizeof(Pack<float>) == kVecBytes, "Pack<float> must be exactly 16 bytes");
static_assert(sizeof(Pack<__half>) == kVecBytes, "Pack<__half> must be exactly 16 bytes");
static_assert(sizeof(Pack<__nv_bfloat16>) == kVecBytes, "Pack<__nv_bfloat16> must be exactly 16 bytes");

__device__ __forceinline__ bool is_aligned16(const void* p) {
  return (reinterpret_cast<uintptr_t>(p) & (kVecBytes - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Reductions.
// ---------------------------------------------------------------------------

// Sum across the 32 lanes of a warp. After the loop only lane 0 holds the
// full sum (lanes >= 32 - offset read garbage from __shfl_down_sync, but
// those lanes' values are never used).
//
//   offset 16: lane i += lane i+16   (lanes 0..15 now hold pair sums)
//   offset  8: lane i += lane i+8    (lanes 0..7 hold sums of 4)
//   offset  4: lane i += lane i+4    (lanes 0..3 hold sums of 8)
//   offset  2: lane i += lane i+2    (lanes 0..1 hold sums of 16)
//   offset  1: lane 0 += lane 1      (lane 0 holds the sum of 32)
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

// Sum across the whole block. On return only thread 0 holds the full sum.
// `partials` needs one float per warp (at most 32 with a 1024-thread block).
// Contains exactly one __syncthreads().
__device__ __forceinline__ float block_reduce_sum(float v, float* partials) {
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int num_warps = blockDim.x / kWarpSize;  // launcher guarantees blockDim.x % 32 == 0

  // Step 1: 32 lanes -> lane 0 of each warp, register to register.
  v = warp_reduce_sum(v);

  // Step 2: lane 0 of every warp parks its partial in shared memory.
  if (lane == 0) {
    partials[warp] = v;
  }
  __syncthreads();

  // Step 3: warp 0 picks up the per-warp partials (lanes beyond num_warps
  // contribute zero) and shuffles once more. Lane 0 of warp 0 == thread 0.
  if (warp == 0) {
    v = (lane < num_warps) ? partials[lane] : 0.0f;
    v = warp_reduce_sum(v);
  }
  return v;  // meaningful only for threadIdx.x == 0
}

// ---------------------------------------------------------------------------
// The kernel.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_kernel(
    const T* __restrict__ x,  // (M, N), row-major, contiguous
    const T* __restrict__ w,  // (N,), contiguous
    T* __restrict__ y,        // (M, N), row-major, contiguous
    int N,
    float eps) {
  using D = DType<T>;
  constexpr int VEC = D::kVec;

  __shared__ float s_partials[kWarpSize];  // one slot per warp; blockDim <= 1024 -> <= 32 warps
  __shared__ float s_rstd;                 // broadcast slot for the row's rsqrt

  const int64_t row = blockIdx.x;
  const T* x_row = x + row * static_cast<int64_t>(N);
  T* y_row = y + row * static_cast<int64_t>(N);

  // Block-uniform decision: every thread of this block looks at the same
  // row, so the branch below never diverges inside a warp. If any of the
  // three pointers is not 16-byte aligned, n_vec == 0 and every element goes
  // through the scalar loops.
  const bool vectorized = is_aligned16(x_row) && is_aligned16(y_row) && is_aligned16(w);
  const int n_vec = vectorized ? N / VEC : 0;  // number of full 16-byte chunks in the row
  const int tail_start = n_vec * VEC;          // first element handled by the scalar loops

  // ---- Pass 1: per-thread partial sum of squares, accumulated in fp32 ----
  // Thread t owns chunks t, t + blockDim, t + 2 * blockDim, ... so that the
  // 32 threads of a warp touch 32 consecutive 16-byte chunks = 512
  // contiguous bytes per load instruction (perfectly coalesced).
  float ss = 0.0f;
  {
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
      Pack<T> p;
      p.raw = x_vec[i];  // one 128-bit load
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        const float f = D::to_float(p.elems[j]);
        ss += f * f;
      }
    }
    for (int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
      const float f = D::to_float(x_row[i]);
      ss += f * f;
    }
  }

  // ---- Reduce across the block, rsqrt in thread 0, broadcast ----
  const float total = block_reduce_sum(ss, s_partials);
  if (threadIdx.x == 0) {
    s_rstd = rsqrtf(total / static_cast<float>(N) + eps);
  }
  __syncthreads();
  const float rstd = s_rstd;

  // ---- Pass 2: y = x * rstd * w, same chunk ownership as pass 1 ----
  // The second read of x is served from L1/L2 for the row sizes this kernel
  // is used at (a few KB per row); HBM sees x once and y once.
  {
    const float4* x_vec = reinterpret_cast<const float4*>(x_row);
    const float4* w_vec = reinterpret_cast<const float4*>(w);
    float4* y_vec = reinterpret_cast<float4*>(y_row);
    for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
      Pack<T> px, pw, py;
      px.raw = x_vec[i];
      pw.raw = w_vec[i];
#pragma unroll
      for (int j = 0; j < VEC; ++j) {
        py.elems[j] = D::from_float(D::to_float(px.elems[j]) * rstd * D::to_float(pw.elems[j]));
      }
      y_vec[i] = py.raw;  // one 128-bit store
    }
    for (int i = tail_start + threadIdx.x; i < N; i += blockDim.x) {
      y_row[i] = D::from_float(D::to_float(x_row[i]) * rstd * D::to_float(w[i]));
    }
  }
}

// ---------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------

// One 16-byte chunk per thread when the row allows it: start at 4 warps and
// double until every chunk has a thread or we hit 1024. Always a power of
// two in [128, 1024], hence a multiple of 32 and at most 32 warps.
int choose_block_size(int64_t N, int vec) {
  const int64_t chunks = (N + vec - 1) / vec;
  int threads = kMinThreadsPerBlock;
  while (threads < chunks && threads < kMaxThreadsPerBlock) {
    threads *= 2;
  }
  return threads;
}

template <typename T>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, double eps) {
  const int64_t M = x.size(0);
  const int64_t N = x.size(1);
  const int threads = choose_block_size(N, DType<T>::kVec);
  const dim3 grid(static_cast<unsigned int>(M));
  const dim3 block(static_cast<unsigned int>(threads));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  rmsnorm_kernel<T><<<grid, block, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()),
      reinterpret_cast<const T*>(w.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()),
      static_cast<int>(N),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  TORCH_CHECK(x.is_cuda(), "rmsnorm_cuda: x must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "rmsnorm_cuda: weight must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "rmsnorm_cuda: x must be 2-D (M, N); the Python wrapper flattens leading dims");
  TORCH_CHECK(x.is_contiguous(), "rmsnorm_cuda: x must be contiguous");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
              "rmsnorm_cuda: weight must have shape (N,) with N = x.size(1)");
  TORCH_CHECK(weight.is_contiguous(), "rmsnorm_cuda: weight must be contiguous");
  TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "rmsnorm_cuda: weight dtype must match x dtype");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(), "rmsnorm_cuda: N does not fit in int32");
  TORCH_CHECK(x.size(0) <= 0x7fffffffLL, "rmsnorm_cuda: M exceeds the maximum grid size");

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(x));
  torch::Tensor y = torch::empty_like(x);
  if (x.numel() == 0) {
    return y;  // nothing to do, and a zero-size grid is not a valid launch
  }

  switch (x.scalar_type()) {
    case at::ScalarType::Half:
      launch<__half>(x, weight, y, eps);
      break;
    case at::ScalarType::BFloat16:
      launch<__nv_bfloat16>(x, weight, y, eps);
      break;
    case at::ScalarType::Float:
      launch<float>(x, weight, y, eps);
      break;
    default:
      TORCH_CHECK(false, "rmsnorm_cuda: unsupported dtype ", x.scalar_type(),
                  "; expected float16, bfloat16 or float32");
  }
  return y;
}
