// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Andrej Karpathy
//
// Independent implementation from published KDA WY/UT equations.  Behavior
// was cross-checked against the pinned MIT semantic references documented by
// the project (FLA a3edffc3, FlashKDA 1ce47ea3, Lethe e3ed0ccb).

#include "chunk_wy_common.cuh"

#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace {

constexpr int kBatch = 2;
constexpr int kHeads = 3;
constexpr int kLength = 4096;
constexpr int kDim = 128;
constexpr int kChunk = 64;
constexpr int kMatrixTile = 16;
constexpr int kChunks = kLength / kChunk;
constexpr int kRecurrences = kBatch * kHeads;
constexpr int kChunkRows = kRecurrences * kChunks;

__device__ __forceinline__ float wy_sigmoid(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

__device__ __forceinline__ int64_t input_vector_index(
    int b, int token, int h, int d) {
  return (((static_cast<int64_t>(b) * kLength + token) * kHeads + h) *
          kDim + d);
}

__device__ __forceinline__ int64_t input_scalar_index(
    int b, int token, int h) {
  return ((static_cast<int64_t>(b) * kLength + token) * kHeads + h);
}

__device__ __forceinline__ int64_t chunk_vector_index(
    int n, int row, int d) {
  return ((static_cast<int64_t>(n) * kChunk + row) * kDim + d);
}

__device__ __forceinline__ int64_t chunk_matrix_index(
    int n, int row, int column) {
  return ((static_cast<int64_t>(n) * kChunk + row) * kChunk + column);
}

// Q retains its original FP32 allocation stride after becoming a compact BF16
// scan view. The doubled BF16 stride keeps every chunk inside its own backing
// range. Restored keys are packed key-major so each E^T WMMA tile is contiguous
// and can be moved by cp.async without an intermediate transpose.
__device__ __forceinline__ int64_t packed_strided_transpose_index(
    int n, int d, int row) {
  return static_cast<int64_t>(n) * kChunk * kDim * 2 + d * kChunk + row;
}

// A warp copies one contiguous 16x16 BF16 tile with 32 aligned 16-byte
// transactions. Each lane owns one half-row; the pipeline group is committed
// and waited by the caller so the next tile can overlap state staging + MMA.
__device__ __forceinline__ void wy_async_copy_bf16_tile(
    __nv_bfloat16* destination,
    const __nv_bfloat16* source,
    int source_stride,
    int lane) {
  const int row = lane / 2;
  const int half = lane - row * 2;
  __pipeline_memcpy_async(
      destination + row * kMatrixTile + half * 8,
      source + row * source_stride + half * 8,
      16);
}

// F0: one key lane follows one channel through a complete 64-token chunk.
// Lane zero performs every norm reduction in ascending-key order.
__global__ void nanochat_kda_wy_preprocess_c64_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    float* qbar,
    float* khat,
    __nv_bfloat16* qgamma,
    float* prefix_g,
    float* beta,
    float* P,
    float* Q,
    float lower_bound,
    float scale) {
  const int n = blockIdx.x;
  const int d = threadIdx.x;
  if (n >= kChunkRows || d >= kDim) {
    return;
  }
  const int chunk_id = n % kChunks;
  const int recurrence = n / kChunks;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token_start = chunk_id * kChunk;

  __shared__ float q_squares[kDim];
  __shared__ float k_squares[kDim];
  __shared__ float q_inverse;
  __shared__ float k_inverse;
  __shared__ float beta_value;

  const float a = expf(A_log[h]);
  float running_g = 0.0f;
  for (int row = 0; row < kChunk; ++row) {
    const int token = token_start + row;
    const int64_t source = input_vector_index(b, token, h, d);
    const float q_value = __bfloat162float(q[source]);
    const float k_value = __bfloat162float(k[source]);
    q_squares[d] = q_value * q_value;
    k_squares[d] = k_value * k_value;
    __syncthreads();

    if (d == 0) {
      float q_sum = 0.0f;
      float k_sum = 0.0f;
      for (int reduction = 0; reduction < kDim; ++reduction) {
        q_sum += q_squares[reduction];
        k_sum += k_squares[reduction];
      }
      q_inverse = rsqrtf(fmaxf(q_sum, 1.0e-24f));
      k_inverse = rsqrtf(fmaxf(k_sum, 1.0e-24f));
      beta_value = wy_sigmoid(__bfloat162float(
          beta_logits[input_scalar_index(b, token, h)]));
      beta[static_cast<int64_t>(n) * kChunk + row] = beta_value;
    }
    __syncthreads();

    const int64_t destination = chunk_vector_index(n, row, d);
    const float normalized_q = (q_value * q_inverse) * scale;
    const float normalized_k = k_value * k_inverse;
    const float gate_input = __bfloat162float(raw_gate[source]) +
        dt_bias[h * kDim + d];
    running_g += lower_bound * wy_sigmoid(a * gate_input);
    const float exp_g = expf(running_g);
    qbar[destination] = normalized_q;
    khat[destination] = normalized_k;
    qgamma[destination] = __float2bfloat16_rn(normalized_q * exp_g);
    prefix_g[destination] = running_g;
    P[destination] = beta_value * __bfloat162float(v[source]);
    Q[destination] = beta_value * exp_g * normalized_k;
    __syncthreads();
  }
}

// F1: M is strict lower triangular and A is inclusive lower triangular.
// Every channel dot product uses a fixed ascending order and the stable
// exp(G_i-G_s) ratio; exp(-G) is never formed.
__global__ void nanochat_kda_wy_build_m_a_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    float* M,
    float* A) {
  const int n = blockIdx.x;
  if (n >= kChunkRows) {
    return;
  }
  for (int pair = threadIdx.x; pair < kChunk * kChunk;
       pair += blockDim.x) {
    const int i = pair / kChunk;
    const int s = pair - i * kChunk;
    float m_value = 0.0f;
    float a_value = 0.0f;
    if (s <= i) {
      for (int d = 0; d < kDim; ++d) {
        const int64_t i_offset = chunk_vector_index(n, i, d);
        const int64_t s_offset = chunk_vector_index(n, s, d);
        const float ratio = expf(prefix_g[i_offset] - prefix_g[s_offset]);
        a_value += qbar[i_offset] * khat[s_offset] * ratio;
        if (s < i) {
          m_value += khat[i_offset] * khat[s_offset] * ratio;
        }
      }
      if (s < i) {
        m_value *= beta[static_cast<int64_t>(n) * kChunk + i];
      }
    }
    const int64_t destination = chunk_matrix_index(n, i, s);
    M[destination] = m_value;
    A[destination] = a_value;
  }
}

// Build one stable 16x16 time tile as two transformed dot products.  The
// per-channel center lies between every target/source pair, so neither
// exponent can span more than 15 frozen lower-bound steps on a diagonal tile;
// both operands are non-growing for strictly lower off-diagonal tiles.
__global__ void nanochat_kda_wy_transform_pair_c64_kernel(
    const float* left_source,
    const float* khat,
    const float* prefix_g,
    float* q_left,
    float* k_left,
    float* right,
    int target_start,
    int source_start) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kElements = kChunkRows * kMatrixTile * kDim;
  if (index >= kElements) {
    return;
  }
  const int n = index / (kMatrixTile * kDim);
  const int within = index - n * kMatrixTile * kDim;
  const int local_row = within / kDim;
  const int d = within - local_row * kDim;
  const int target_row = target_start + local_row;
  const int source_row = source_start + local_row;
  const int center_row =
      target_start == source_start ? target_start : target_start - 1;
  const float center = prefix_g[chunk_vector_index(n, center_row, d)];
  const int64_t target = chunk_vector_index(n, target_row, d);
  const int64_t source = chunk_vector_index(n, source_row, d);
  const float target_factor = expf(prefix_g[target] - center);
  q_left[index] = left_source[target] * target_factor;
  k_left[index] = khat[target] * target_factor;
  right[index] = khat[source] * expf(center - prefix_g[source]);
}

// Build one stable A/M tile pair in a single CTA. The pack is shared by both
// products, BF16 is explicit at the WMMA boundary, and FP32 accumulators are
// masked/scaled into the final 64x64 matrices without scratch tensors.
__global__ void nanochat_kda_wy_build_pair_wmma_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    float* M,
    __nv_bfloat16* A) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTileCount = kChunk / kMatrixTile;
  constexpr int kPairCount = kTileCount * (kTileCount + 1) / 2;
  const int pair = blockIdx.x / kChunkRows;
  const int n = blockIdx.x - pair * kChunkRows;
  if (pair >= kPairCount || n >= kChunkRows) {
    return;
  }
  int target_tile = 0;
  int source_tile = pair;
  while (source_tile > target_tile) {
    source_tile -= target_tile + 1;
    ++target_tile;
  }
  const int target_start = target_tile * kMatrixTile;
  const int source_start = source_tile * kMatrixTile;
  constexpr int kTileElements = kMatrixTile * kDim;
  __shared__ __nv_bfloat16 left[2 * kTileElements];
  __shared__ __nv_bfloat16 right[kTileElements];
  __shared__ float product[2 * kMatrixTile * kMatrixTile];

  for (int index = threadIdx.x; index < kTileElements;
       index += blockDim.x) {
    const int local_row = index / kDim;
    const int d = index - local_row * kDim;
    const int target_row = target_start + local_row;
    const int source_row = source_start + local_row;
    const int center_row = target_start == source_start
        ? target_start : target_start - 1;
    const int64_t target = chunk_vector_index(n, target_row, d);
    const int64_t source = chunk_vector_index(n, source_row, d);
    const float center = prefix_g[chunk_vector_index(n, center_row, d)];
    const float target_factor = expf(prefix_g[target] - center);
    left[index] = __float2bfloat16_rn(qbar[target] * target_factor);
    left[kTileElements + index] =
        __float2bfloat16_rn(khat[target] * target_factor);
    right[index] =
        __float2bfloat16_rn(khat[source] * expf(center - prefix_g[source]));
  }
  __syncthreads();

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::col_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kMatrixTile, kMatrixTile, kMatrixTile, float>;
  const int warp = threadIdx.x / warpSize;
  if (warp < 2) {
    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int key_start = 0; key_start < kDim; key_start += kMatrixTile) {
      MatrixA a;
      MatrixB b;
      wmma::load_matrix_sync(
          a, left + warp * kTileElements + key_start, kDim);
      wmma::load_matrix_sync(b, right + key_start, kDim);
      wmma::mma_sync(accumulator, a, b, accumulator);
    }
    wmma::store_matrix_sync(
        product + warp * kMatrixTile * kMatrixTile, accumulator,
        kMatrixTile, wmma::mem_row_major);
  }
  __syncthreads();

  const int index = threadIdx.x;
  if (index < kMatrixTile * kMatrixTile) {
    const int local_row = index / kMatrixTile;
    const int local_source = index - local_row * kMatrixTile;
    const int row = target_start + local_row;
    const int source = source_start + local_source;
    const int64_t destination = chunk_matrix_index(n, row, source);
    A[destination] = __float2bfloat16_rn(
        source <= row ? product[index] : 0.0f);
    M[destination] = source < row
        ? beta[static_cast<int64_t>(n) * kChunk + row] *
            product[kMatrixTile * kMatrixTile + index]
        : 0.0f;
  }
}

__global__ void nanochat_kda_wy_transform_left_k_c64_kernel(
    const float* khat,
    const float* prefix_g,
    float* left,
    int target_start,
    int source_start) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kElements = kChunkRows * kMatrixTile * kDim;
  if (index >= kElements) {
    return;
  }
  const int n = index / (kMatrixTile * kDim);
  const int within = index - n * kMatrixTile * kDim;
  const int local_row = within / kDim;
  const int d = within - local_row * kDim;
  const int target_row = target_start + local_row;
  const int center_row =
      target_start == source_start ? target_start : target_start - 1;
  const int64_t target = chunk_vector_index(n, target_row, d);
  const float center = prefix_g[chunk_vector_index(n, center_row, d)];
  left[index] = khat[target] * expf(prefix_g[target] - center);
}

__global__ void nanochat_kda_wy_finish_m_a_c64_kernel(
    const float* beta,
    float* M,
    float* A) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kElements = kChunkRows * kChunk * kChunk;
  if (index >= kElements) {
    return;
  }
  const int n = index / (kChunk * kChunk);
  const int within = index - n * kChunk * kChunk;
  const int row = within / kChunk;
  const int source = within - row * kChunk;
  if (source > row) {
    A[index] = 0.0f;
  }
  if (source >= row) {
    M[index] = 0.0f;
  } else {
    M[index] *= beta[static_cast<int64_t>(n) * kChunk + row];
  }
}

__global__ void nanochat_kda_wy_rebuild_p_c64_kernel(
    const __nv_bfloat16* v,
    const float* beta,
    float* P) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kElements = kChunkRows * kChunk * kDim;
  if (index >= kElements) {
    return;
  }
  const int n = index / (kChunk * kDim);
  const int within = index - n * kChunk * kDim;
  const int row = within / kDim;
  const int d = within - row * kDim;
  const int chunk_id = n % kChunks;
  const int recurrence = n / kChunks;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token = chunk_id * kChunk + row;
  P[index] = beta[static_cast<int64_t>(n) * kChunk + row] *
      __bfloat162float(v[input_vector_index(b, token, h, d)]);
}

// F2: columns of (I+M)^-1 are independent once preceding rows are ready.
// The row loop and each ascending inner product are deliberately fixed order.
__global__ void nanochat_kda_wy_unit_lower_solve_c64_kernel(
    const float* M,
    float* T) {
  const int n = blockIdx.x;
  const int column = threadIdx.x;
  if (n >= kChunkRows || column >= kChunk) {
    return;
  }
  for (int row = 0; row < kChunk; ++row) {
    float value = 0.0f;
    if (column == row) {
      value = 1.0f;
    } else if (column < row) {
      float sum = 0.0f;
      for (int inner = column; inner < row; ++inner) {
        sum += M[chunk_matrix_index(n, row, inner)] *
            T[chunk_matrix_index(n, inner, column)];
      }
      value = -sum;
    }
    T[chunk_matrix_index(n, row, column)] = value;
    __syncthreads();
  }
}

// F4: each CTA owns one value row of the K-first state.  It scans chunks in
// order and performs every key reduction and 64-row product in fixed order.
__global__ void nanochat_kda_chunk_wy_forward_simt_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* A,
    const float* U,
    const float* W,
    __nv_bfloat16* output) {
  const int owner = blockIdx.x;
  const int d = threadIdx.x;
  if (owner >= kBatch * kHeads * kDim || d >= kDim) {
    return;
  }
  const int value = owner % kDim;
  const int recurrence = owner / kDim;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;

  __shared__ float contribution[kDim];
  __shared__ float z[kChunk];
  float state = 0.0f;

  for (int chunk_id = 0; chunk_id < kChunks; ++chunk_id) {
    const int n = recurrence * kChunks + chunk_id;
    const int token_start = chunk_id * kChunk;
    for (int row = 0; row < kChunk; ++row) {
      const int64_t vector_offset = chunk_vector_index(n, row, d);
      contribution[d] = W[vector_offset] * state;
      __syncthreads();
      if (d == 0) {
        float sum = 0.0f;
        for (int key = 0; key < kDim; ++key) {
          sum += contribution[key];
        }
        z[row] = U[chunk_vector_index(n, row, value)] - sum;
      }
      __syncthreads();

      contribution[d] =
          (qbar[vector_offset] * expf(prefix_g[vector_offset])) * state;
      __syncthreads();
      if (d == 0) {
        float output_value = 0.0f;
        for (int key = 0; key < kDim; ++key) {
          output_value += contribution[key];
        }
        for (int source_row = 0; source_row <= row; ++source_row) {
          output_value += A[chunk_matrix_index(n, row, source_row)] *
              z[source_row];
        }
        const int token = token_start + row;
        output[input_vector_index(b, token, h, value)] =
            __float2bfloat16_rn(output_value);
      }
      __syncthreads();
    }

    const float end_g = prefix_g[chunk_vector_index(n, kChunk - 1, d)];
    float next_state = expf(end_g) * state;
    for (int row = 0; row < kChunk; ++row) {
      const int64_t vector_offset = chunk_vector_index(n, row, d);
      next_state += khat[vector_offset] *
          expf(end_g - prefix_g[vector_offset]) * z[row];
    }
    state = next_state;
    __syncthreads();
  }
}

// F4 scan pack: form the two decayed boundary operands for one chunk in a
// dense recurrence-major batch.  The subsequent four products are FP32 ATen
// batched matrix multiplies; this project kernel keeps the scan dataflow and
// profiler provenance explicit without changing the validated F0--F3 stages.
__global__ void nanochat_kda_wy_scan_pack_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    float* q_decay,
    float* end_decay_k,
    int chunk_id) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int recurrence = index / (kChunk * kDim);
  const int within = index - recurrence * kChunk * kDim;
  const int row = within / kDim;
  const int d = within - row * kDim;
  const int n = recurrence * kChunks + chunk_id;
  const int64_t source = chunk_vector_index(n, row, d);
  const float g = prefix_g[source];
  const float g_end =
      prefix_g[chunk_vector_index(n, kChunk - 1, d)];
  q_decay[index] = qbar[source] * expf(g);
  end_decay_k[index] = khat[source] * expf(g_end - g);
}

// Complete Z = U-WH without an allocating tensor expression.
__global__ void nanochat_kda_wy_scan_z_c64_kernel(
    const float* U,
    const float* wh,
    float* z,
    int chunk_id) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int recurrence = index / (kChunk * kDim);
  const int within = index - recurrence * kChunk * kDim;
  const int n = recurrence * kChunks + chunk_id;
  z[index] = U[static_cast<int64_t>(n) * kChunk * kDim + within] - wh[index];
}

// Write O = q_decay H + AZ in the frozen [B,T,H,V] layout and apply the
// diagonal boundary decay to H in place before the E^T Z update.
__global__ void nanochat_kda_wy_scan_output_decay_c64_kernel(
    const float* qh,
    const float* az,
    const float* prefix_g,
    float* state,
    __nv_bfloat16* output,
    int chunk_id) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kOutputElements = kRecurrences * kChunk * kDim;
  constexpr int kStateElements = kRecurrences * kDim * kDim;
  if (index < kOutputElements) {
    const int recurrence = index / (kChunk * kDim);
    const int within = index - recurrence * kChunk * kDim;
    const int row = within / kDim;
    const int value = within - row * kDim;
    const int b = recurrence / kHeads;
    const int h = recurrence - b * kHeads;
    const int token = chunk_id * kChunk + row;
    output[input_vector_index(b, token, h, value)] =
        __float2bfloat16_rn(qh[index] + az[index]);
  }
  if (index < kStateElements) {
    const int recurrence = index / (kDim * kDim);
    const int within = index - recurrence * kDim * kDim;
    const int key = within / kDim;
    const int n = recurrence * kChunks + chunk_id;
    const float g_end =
        prefix_g[chunk_vector_index(n, kChunk - 1, key)];
    state[index] *= expf(g_end);
  }
}

__global__ void nanochat_kda_wy_scan_state_add_c64_kernel(
    float* state,
    const float* delta) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kElements = kRecurrences * kDim * kDim;
  if (index < kElements) {
    state[index] += delta[index];
  }
}

// Q/T are dead after U/W are formed. Convert only the restored key and W into
// those existing allocations: preprocess already emitted qgamma, and the pair
// builder wrote A directly at its scan precision.
__global__ void nanochat_kda_wy_pack_async_scan_vectors_c64_kernel(
    const float* khat,
    const float* prefix_g,
    const float* W,
    float* restored_storage,
    float* W_storage) {
  constexpr int kElements = kChunkRows * kChunk * kDim;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kElements) {
    return;
  }
  constexpr int kChunkElements = kChunk * kDim;
  const int n = index / kChunkElements;
  const int within = index - n * kChunkElements;
  const int row = within / kDim;
  const int key = within - row * kDim;
  const int64_t source = chunk_vector_index(n, row, key);
  const float end_g = prefix_g[chunk_vector_index(n, kChunk - 1, key)];
  auto* packed_k = reinterpret_cast<__nv_bfloat16*>(restored_storage);
  auto* packed_W = reinterpret_cast<__nv_bfloat16*>(W_storage);
  packed_k[packed_strided_transpose_index(n, key, row)] =
      __float2bfloat16_rn(khat[source] * expf(end_g - prefix_g[source]));
  packed_W[index] = __float2bfloat16_rn(W[source]);
}

// F4 tensor-core scan: one persistent CTA owns 32 adjacent V rows for one
// (batch, head) recurrence.  A 32-value tile keeps the 24 CTAs large enough
// to cover GB10 while sharing each W/qgamma/E master-operand load across two
// 16-column WMMA products.  The 48 KiB shared working set permits the FP32
// state, residual, output/update accumulators, and explicit BF16 cast tiles to
// remain CTA-local without a workspace allocation.
__global__ void nanochat_kda_chunk_wy_forward_wmma_c64_kernel(
    const __nv_bfloat16* qgamma,
    const __nv_bfloat16* restored_k,
    const float* prefix_g,
    const __nv_bfloat16* A,
    const float* U,
    const __nv_bfloat16* W,
    __nv_bfloat16* output,
    float* group_state_checkpoints) {
  namespace wmma = nvcuda::wmma;
  constexpr int kValueTile = 32;
  constexpr int kMatrix = 16;
  constexpr int kWarps = 8;
  constexpr int kTileElements = kMatrix * kMatrix;

  const int owner = blockIdx.x;
  if (owner >= kBatch * kHeads * (kDim / kValueTile)) {
    return;
  }
  const int value_tile = owner % (kDim / kValueTile);
  const int recurrence = owner / (kDim / kValueTile);
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int value_start = value_tile * kValueTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  // H and next_state use key-major, V-tile-minor storage.  Public tensors keep
  // their required V-first indexing; the transposition exists only inside the
  // CTA so row-major WMMA B tiles have unit-stride value columns.
  __shared__ float state[kDim * kValueTile];
  __shared__ float z[kChunk * kValueTile];
  __shared__ __align__(16) float next_state[kDim * kValueTile];
  __shared__ __align__(16) __nv_bfloat16
      operand_a[2 * kWarps * kTileElements];
  // Each phase consumes its BF16 right tile before storing an FP32 result into
  // next_state. Reuse the first 4 KiB across that explicit lifetime boundary,
  // keeping double-buffered A tiles within the 48-KiB static shared limit.
  __nv_bfloat16* operand_b =
      reinterpret_cast<__nv_bfloat16*>(next_state);

  for (int index = threadIdx.x; index < kDim * kValueTile;
       index += blockDim.x) {
    state[index] = 0.0f;
  }
  __syncthreads();

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kMatrix, kMatrix, kMatrix,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kMatrix, kMatrix, kMatrix,
      __nv_bfloat16, wmma::row_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kMatrix, kMatrix, kMatrix, float>;

  for (int chunk_id = 0; chunk_id < kChunks; ++chunk_id) {
    const int n = recurrence * kChunks + chunk_id;
    const int token_start = chunk_id * kChunk;

    // Z = U - W H.  All eight resident warps own one (16-row, 16-value)
    // product so the two value halves execute concurrently.
    if (warp < kWarps) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int row_start = (warp / 2) * kMatrix;
      const int value_half = warp % 2;
      int stage = 0;
      wy_async_copy_bf16_tile(
          operand_a + warp * kTileElements,
          W + chunk_vector_index(n, row_start, 0), kDim, lane);
      __pipeline_commit();
      for (int key_start = 0; key_start < kDim; key_start += kMatrix) {
        __pipeline_wait_prior(0);
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment,
            operand_a + (stage * kWarps + warp) * kTileElements,
            kMatrix);
        const int next_key = key_start + kMatrix;
        if (next_key < kDim) {
          const int next_stage = stage ^ 1;
          wy_async_copy_bf16_tile(
              operand_a + (next_stage * kWarps + warp) * kTileElements,
              W + chunk_vector_index(n, row_start, next_key), kDim, lane);
          __pipeline_commit();
        }
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const float master = state[
              (key_start + row) * kValueTile +
              value_half * kMatrix + column];
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(master);
        }
        __syncwarp();
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
        stage ^= 1;
      }
      wmma::store_matrix_sync(
          z + row_start * kValueTile + value_half * kMatrix,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kValueTile;
         index += blockDim.x) {
      const int row = index / kValueTile;
      const int value = index - row * kValueTile;
      z[index] = U[chunk_vector_index(n, row, value_start + value)] - z[index];
    }
    __syncthreads();

    // O = qgamma H + A Z.  next_state is reused as a 64x32 FP32 output
    // accumulator here, then overwritten only after public output is stored.
    if (warp < kWarps) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int row_start = (warp / 2) * kMatrix;
      const int value_half = warp % 2;
      int stage = 0;
      wy_async_copy_bf16_tile(
          operand_a + warp * kTileElements,
          qgamma + chunk_vector_index(n, row_start, 0),
          kDim, lane);
      __pipeline_commit();
      for (int key_start = 0; key_start < kDim; key_start += kMatrix) {
        __pipeline_wait_prior(0);
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment,
            operand_a + (stage * kWarps + warp) * kTileElements,
            kMatrix);
        const int next_key = key_start + kMatrix;
        if (next_key < kDim) {
          const int next_stage = stage ^ 1;
          wy_async_copy_bf16_tile(
              operand_a + (next_stage * kWarps + warp) * kTileElements,
              qgamma + chunk_vector_index(n, row_start, next_key),
              kDim, lane);
          __pipeline_commit();
        }
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const float master = state[
              (key_start + row) * kValueTile +
              value_half * kMatrix + column];
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(master);
        }
        __syncwarp();
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
        stage ^= 1;
      }
      stage = 0;
      wy_async_copy_bf16_tile(
          operand_a + warp * kTileElements,
          A + chunk_matrix_index(n, row_start, 0), kChunk, lane);
      __pipeline_commit();
      for (int source_start = 0; source_start < kChunk;
           source_start += kMatrix) {
        __pipeline_wait_prior(0);
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment,
            operand_a + (stage * kWarps + warp) * kTileElements,
            kMatrix);
        const int next_source = source_start + kMatrix;
        if (next_source < kChunk) {
          const int next_stage = stage ^ 1;
          wy_async_copy_bf16_tile(
              operand_a + (next_stage * kWarps + warp) * kTileElements,
              A + chunk_matrix_index(n, row_start, next_source),
              kChunk, lane);
          __pipeline_commit();
        }
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const float master = z[
              (source_start + row) * kValueTile +
              value_half * kMatrix + column];
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(master);
        }
        __syncwarp();
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
        stage ^= 1;
      }
      // operand_b aliases next_state across warps. All right fragments must be
      // register-resident before any warp overwrites that shared region.
      __syncthreads();
      wmma::store_matrix_sync(
          next_state + row_start * kValueTile + value_half * kMatrix,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kValueTile;
         index += blockDim.x) {
      const int row = index / kValueTile;
      const int value = index - row * kValueTile;
      output[input_vector_index(
          b, token_start + row, h, value_start + value)] =
          __float2bfloat16_rn(next_state[index]);
    }
    __syncthreads();

    // Hnext = decay H + E^T Z.  Eight warps own the eight key-row tiles and
    // share each explicitly cast E^T fragment across both value halves.
    if (warp < kDim / kMatrix) {
      Accumulator accumulators[2];
      wmma::fill_fragment(accumulators[0], 0.0f);
      wmma::fill_fragment(accumulators[1], 0.0f);
      const int key_start = warp * kMatrix;
      int stage = 0;
      wy_async_copy_bf16_tile(
          operand_a + warp * kTileElements,
          restored_k + packed_strided_transpose_index(n, key_start, 0),
          kChunk, lane);
      __pipeline_commit();
      for (int source_start = 0; source_start < kChunk;
           source_start += kMatrix) {
        __pipeline_wait_prior(0);
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment,
            operand_a + (stage * kWarps + warp) * kTileElements,
            kMatrix);
        const int next_source = source_start + kMatrix;
        if (next_source < kChunk) {
          const int next_stage = stage ^ 1;
          wy_async_copy_bf16_tile(
              operand_a + (next_stage * kWarps + warp) * kTileElements,
              restored_k + packed_strided_transpose_index(
                  n, key_start, next_source),
              kChunk, lane);
          __pipeline_commit();
        }
        for (int value_half = 0; value_half < 2; ++value_half) {
          for (int element = lane; element < kTileElements;
               element += warpSize) {
            const int row = element / kMatrix;
            const int column = element - row * kMatrix;
            const float master = z[
                (source_start + row) * kValueTile +
                value_half * kMatrix + column];
            operand_b[warp * kTileElements + element] =
                __float2bfloat16_rn(master);
          }
          __syncwarp();
          MatrixB b_fragment;
          wmma::load_matrix_sync(
              b_fragment, operand_b + warp * kTileElements, kMatrix);
          wmma::mma_sync(
              accumulators[value_half], a_fragment, b_fragment,
              accumulators[value_half]);
        }
        stage ^= 1;
      }
      // Preserve the same alias lifetime boundary for the Hnext stores.
      __syncthreads();
      wmma::store_matrix_sync(
          next_state + key_start * kValueTile,
          accumulators[0], kValueTile, wmma::mem_row_major);
      wmma::store_matrix_sync(
          next_state + key_start * kValueTile + kMatrix,
          accumulators[1], kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      const int key = index / kValueTile;
      const float end_g = prefix_g[chunk_vector_index(
          n, kChunk - 1, key)];
      state[index] = expf(end_g) * state[index] + next_state[index];
    }
    __syncthreads();

    // Save only inter-group FP32 boundaries in the output allocation's hidden
    // sidecar.  Backward can reconstruct each eight-chunk group's H/Z history
    // with its already-live reverse operands instead of replaying all forward
    // producers a second time.
    if ((chunk_id + 1) % 8 == 0 && chunk_id + 1 < kChunks) {
      const int boundary = chunk_id / 8;
      for (int index = threadIdx.x; index < kDim * kValueTile;
           index += blockDim.x) {
        const int key = index / kValueTile;
        const int value = index - key * kValueTile;
        group_state_checkpoints[
            ((static_cast<int64_t>(boundary) * kRecurrences + recurrence) *
                 kDim + key) * kDim + value_start + value] = state[index];
      }
      __syncthreads();
    }
  }
}

}  // namespace

at::Tensor nanochat_kda_chunk_wy_forward_c64(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    float lower_bound,
    float scale) {
  const at::TensorOptions fp32 = A_log.options();
  at::Tensor qbar = at::empty({kChunkRows, kChunk, kDim}, fp32);
  at::Tensor khat = at::empty_like(qbar);
  at::Tensor qgamma = at::empty(
      {kChunkRows, kChunk, kDim}, q.options());
  at::Tensor prefix_g = at::empty_like(qbar);
  at::Tensor beta = at::empty({kChunkRows, kChunk}, fp32);
  at::Tensor P = at::empty_like(qbar);
  at::Tensor Q = at::empty_like(qbar);
  at::Tensor M = at::empty({kChunkRows, kChunk, kChunk}, fp32);
  at::Tensor A = at::empty(
      {kChunkRows, kChunk, kChunk}, q.options());
  at::Tensor T = at::empty_like(M);
  constexpr int64_t kOutputElements =
      static_cast<int64_t>(kBatch) * kLength * kHeads * kDim;
  constexpr int64_t kCheckpointElements =
      static_cast<int64_t>(kChunks / 8 - 1) * kRecurrences * kDim * kDim;
  // The visible tensor is a contiguous prefix.  Its saved storage keeps seven
  // FP32 group-boundary states alive until the protected backward receives the
  // exact same output tensor, without changing the public shape or dtype.
  at::Tensor output_storage = at::empty(
      {kOutputElements + 2 * kCheckpointElements}, v.options());
  at::Tensor output = output_storage.narrow(0, 0, kOutputElements).view(
      {kBatch, kLength, kHeads, kDim});
  float* group_state_checkpoints = reinterpret_cast<float*>(
      reinterpret_cast<__nv_bfloat16*>(
          output_storage.data_ptr<at::BFloat16>()) + kOutputElements);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  constexpr int kThreads = 256;
  at::NoGradGuard no_grad;
  at::NoTF32Guard no_tf32;
  // Pair-builder launches cover only the lower 16x16 tile triangle, while the
  // persistent WMMA scan loads all four source tiles for every output row.
  // Define the untouched upper tiles explicitly instead of depending on the
  // contents returned by the caching allocator.
  C10_CUDA_CHECK(cudaMemsetAsync(
      A.data_ptr<at::BFloat16>(), 0,
      static_cast<size_t>(A.numel()) * sizeof(__nv_bfloat16), stream));
  nanochat_kda_wy_preprocess_c64_kernel<<<
      kChunkRows, kDim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          raw_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          beta_logits.data_ptr<at::BFloat16>()),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      qbar.data_ptr<float>(), khat.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(qgamma.data_ptr<at::BFloat16>()),
      prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
      P.data_ptr<float>(), Q.data_ptr<float>(), lower_bound, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  constexpr int kPairCount =
      (kChunk / kMatrixTile) * (kChunk / kMatrixTile + 1) / 2;
  nanochat_kda_wy_build_pair_wmma_c64_kernel<<<
      kPairCount * kChunkRows, kThreads, 0, stream>>>(
      qbar.data_ptr<float>(), khat.data_ptr<float>(),
      prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
      M.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(A.data_ptr<at::BFloat16>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  nanochat_kda_wy_unit_lower_solve_c64_kernel<<<
      kChunkRows, kChunk, 0, stream>>>(M.data_ptr<float>(), T.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // ATen uses the PyTorch allocator and the current stream for these two
  // homogeneous FP32 products.  No private handle or hidden allocation exists.
  // Explicit guards prevent autograd recording and TF32 contraction inside the
  // custom-autograd operator.
  at::Tensor U = at::empty_like(P);
  at::Tensor W = at::empty_like(Q);
  at::bmm_out(U, T, P);
  at::bmm_out(W, T, Q);

  constexpr int kVectorElements = kChunkRows * kChunk * kDim;
  nanochat_kda_wy_pack_async_scan_vectors_c64_kernel<<<
      (kVectorElements + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      khat.data_ptr<float>(), prefix_g.data_ptr<float>(), W.data_ptr<float>(),
      Q.data_ptr<float>(), T.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // The tensor-core scan owns 32 adjacent value rows per CTA. It retains FP32
  // state/accumulators and double-buffers aligned BF16 left-operand tiles.
  nanochat_kda_chunk_wy_forward_wmma_c64_kernel<<<
      kBatch * kHeads * (kDim / 32), 256, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(
          qgamma.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr<float>()),
      prefix_g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
      U.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(T.data_ptr<float>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      group_state_checkpoints);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
