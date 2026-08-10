// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Andrej Karpathy
//
// Independently derived reverse-mode implementation of the published C64
// KDA WY/UT equations.  The matrix VJP is checked against an FP64 multi-chunk
// equation prototype; non-production shapes retain the analytical recurrence.

#include "chunk_wy_common.cuh"

#include <ATen/Context.h>
#include <ATen/core/grad_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda_bf16.h>
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
constexpr int kTilePairs = 10;
constexpr int kStorageGroupChunks = 8;
constexpr int kStorageGroupRows = kRecurrences * kStorageGroupChunks;

__device__ __forceinline__ void causal_tile_pair(
    int pair, int& target_tile, int& source_tile) {
  target_tile = 0;
  while (pair >= target_tile + 1) {
    pair -= target_tile + 1;
    ++target_tile;
  }
  source_tile = pair;
}

// Four conflict-free rounds cover the diagonal loops and the three perfect
// matchings of the off-diagonal K4 tile graph. Pair CTAs in one round touch
// disjoint target/source tiles, so they can update vector gradients directly
// without atomics or a cross-CTA reduction.
__device__ __forceinline__ int colored_causal_pair(int color, int slot) {
  if (color == 0) {
    constexpr int diagonal[4] = {0, 2, 5, 9};
    return diagonal[slot];
  }
  if (color == 1) {
    constexpr int matching[2] = {1, 8};
    return matching[slot];
  }
  if (color == 2) {
    constexpr int matching[2] = {3, 7};
    return matching[slot];
  }
  constexpr int matching[2] = {6, 4};
  return matching[slot];
}

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

// Backward consumes these workspaces exclusively in eight-chunk groups. Store
// them in that order so each reverse group is already a contiguous BMM batch.
__device__ __forceinline__ int grouped_chunk_index(int n) {
  const int chunk_id = n % kChunks;
  const int recurrence = n / kChunks;
  const int group_id = chunk_id / kStorageGroupChunks;
  const int local_chunk = chunk_id % kStorageGroupChunks;
  return group_id * kStorageGroupRows +
      recurrence * kStorageGroupChunks + local_chunk;
}

__device__ __forceinline__ int64_t grouped_vector_index(
    int n, int row, int d) {
  return ((static_cast<int64_t>(grouped_chunk_index(n)) * kChunk + row) *
          kDim + d);
}

__device__ __forceinline__ int64_t grouped_matrix_index(
    int n, int row, int column) {
  return ((static_cast<int64_t>(grouped_chunk_index(n)) * kChunk + row) *
          kChunk + column);
}

__global__ void nanochat_kda_wy_backward_preprocess_c64_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    float* q_inverse,
    float* k_inverse,
    float* qbar,
    float* khat,
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
  __shared__ float q_inv;
  __shared__ float k_inv;
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
      for (int key = 0; key < kDim; ++key) {
        q_sum += q_squares[key];
        k_sum += k_squares[key];
      }
      q_inv = rsqrtf(fmaxf(q_sum, 1.0e-24f));
      k_inv = rsqrtf(fmaxf(k_sum, 1.0e-24f));
      q_inverse[static_cast<int64_t>(n) * kChunk + row] = q_inv;
      k_inverse[static_cast<int64_t>(n) * kChunk + row] = k_inv;
      beta_value = wy_sigmoid(__bfloat162float(
          beta_logits[input_scalar_index(b, token, h)]));
      beta[static_cast<int64_t>(n) * kChunk + row] = beta_value;
    }
    __syncthreads();

    const int64_t destination = chunk_vector_index(n, row, d);
    const float normalized_q = (q_value * q_inv) * scale;
    const float normalized_k = k_value * k_inv;
    const float gate_input = __bfloat162float(raw_gate[source]) +
        dt_bias[h * kDim + d];
    running_g += lower_bound * wy_sigmoid(a * gate_input);
    qbar[destination] = normalized_q;
    khat[destination] = normalized_k;
    prefix_g[destination] = running_g;
    const int64_t grouped_destination = grouped_vector_index(n, row, d);
    P[grouped_destination] = beta_value * __bfloat162float(v[source]);
    Q[grouped_destination] = beta_value * expf(running_g) * normalized_k;
    __syncthreads();
  }
}

__global__ void nanochat_kda_wy_backward_build_m_a_c64_kernel(
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
        const int64_t io = chunk_vector_index(n, i, d);
        const int64_t so = chunk_vector_index(n, s, d);
        const float ratio = expf(prefix_g[io] - prefix_g[so]);
        a_value += qbar[io] * khat[so] * ratio;
        if (s < i) {
          m_value += khat[io] * khat[so] * ratio;
        }
      }
      if (s < i) {
        m_value *= beta[static_cast<int64_t>(n) * kChunk + i];
      }
    }
    M[chunk_matrix_index(n, i, s)] = m_value;
    A[chunk_matrix_index(n, i, s)] = a_value;
  }
}

__global__ void nanochat_kda_wy_backward_transform_pair_c64_kernel(
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

__global__ void nanochat_kda_wy_backward_build_pair_wmma_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    float* M,
    float* A) {
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
    const int64_t m_destination = chunk_matrix_index(n, row, source);
    const int64_t a_destination = grouped_matrix_index(n, row, source);
    A[a_destination] = source <= row ? product[index] : 0.0f;
    M[m_destination] = source < row
        ? beta[static_cast<int64_t>(n) * kChunk + row] *
            product[kMatrixTile * kMatrixTile + index]
        : 0.0f;
  }
}

__global__ void nanochat_kda_wy_backward_transform_left_k_c64_kernel(
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

__global__ void nanochat_kda_wy_backward_finish_m_a_c64_kernel(
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

__global__ void nanochat_kda_wy_backward_rebuild_p_c64_kernel(
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

__global__ void nanochat_kda_wy_backward_solve_c64_kernel(
    const float* M, float* T) {
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
            T[grouped_matrix_index(n, inner, column)];
      }
      value = -sum;
    }
    T[grouped_matrix_index(n, row, column)] = value;
    __syncthreads();
  }
}

__global__ void nanochat_kda_wy_backward_pack_all_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    float* R,
    float* E) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kChunkRows * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int n = index / (kChunk * kDim);
  const int within = index - n * kChunk * kDim;
  const int row = within / kDim;
  const int d = within - row * kDim;
  const int64_t source = chunk_vector_index(n, row, d);
  const float g = prefix_g[source];
  const float end_g = prefix_g[chunk_vector_index(n, kChunk - 1, d)];
  R[index] = qbar[source] * expf(g);
  E[index] = khat[source] * expf(end_g - g);
}

__global__ void nanochat_kda_wy_backward_pack_group_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    float* R,
    float* E,
    int chunk_start,
    int group_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * group_chunks * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int local_n = index / (kChunk * kDim);
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  const int within = index - local_n * kChunk * kDim;
  const int row = within / kDim;
  const int d = within - row * kDim;
  const int64_t source = chunk_vector_index(n, row, d);
  const float g = prefix_g[source];
  const float end_g = prefix_g[chunk_vector_index(n, kChunk - 1, d)];
  R[index] = qbar[source] * expf(g);
  E[index] = khat[source] * expf(end_g - g);
}

__global__ void nanochat_kda_wy_backward_sub_z_c64_kernel(
    const float* U, const float* wh, float* z, int chunk_id) {
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

__global__ void nanochat_kda_wy_backward_sub_group_z_c64_kernel(
    const float* U, const float* wh, float* z,
    int local_chunk, int group_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int recurrence = index / (kChunk * kDim);
  const int within = index - recurrence * kChunk * kDim;
  const int group_n = recurrence * group_chunks + local_chunk;
  z[index] = U[static_cast<int64_t>(group_n) * kChunk * kDim + within] -
      wh[index];
}

__global__ void nanochat_kda_wy_backward_decay_state_c64_kernel(
    const float* prefix_g, float* state, int chunk_id) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * kDim * kDim;
  if (index >= elements) {
    return;
  }
  const int recurrence = index / (kDim * kDim);
  const int within = index - recurrence * kDim * kDim;
  const int key = within / kDim;
  const int n = recurrence * kChunks + chunk_id;
  state[index] *= expf(prefix_g[chunk_vector_index(n, kChunk - 1, key)]);
}

__global__ void nanochat_kda_wy_backward_add_c64_kernel(
    float* destination, const float* source, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] += source[index];
  }
}

// One persistent CTA owns 16 adjacent value columns for one recurrence while
// reconstructing an eight-chunk group boundary. FP32 state and accumulators
// remain CTA-local; only WMMA operands are explicitly rounded to BF16.
__global__ void nanochat_kda_wy_backward_group_boundary_wmma_c64_kernel(
    const float* prefix_g,
    const float* U,
    const float* W,
    const float* E,
    float* state,
    __nv_bfloat16* state_history,
    __nv_bfloat16* z_history,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kValueTile = 16;
  constexpr int kMatrix = 16;
  constexpr int kWarps = 8;
  constexpr int kTileElements = kMatrix * kMatrix;

  const int owner = blockIdx.x;
  if (owner >= kRecurrences * (kDim / kValueTile)) {
    return;
  }
  const int value_tile = owner % (kDim / kValueTile);
  const int recurrence = owner / (kDim / kValueTile);
  const int value_start = value_tile * kValueTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  __shared__ float local_state[kDim * kValueTile];
  __shared__ float z[kChunk * kValueTile];
  __shared__ float next_state[kDim * kValueTile];
  __shared__ __nv_bfloat16 operand_a[kWarps * kTileElements];
  __shared__ __nv_bfloat16 operand_b[kWarps * kTileElements];

  for (int index = threadIdx.x; index < kDim * kValueTile;
       index += blockDim.x) {
    const int key = index / kValueTile;
    const int value = index - key * kValueTile;
    local_state[index] = state[
        (static_cast<int64_t>(recurrence) * kDim + key) * kDim +
        value_start + value];
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

  for (int local_chunk = 0; local_chunk < group_chunks; ++local_chunk) {
    const int group_n = recurrence * group_chunks + local_chunk;
    const int chunk_id = chunk_start + local_chunk;
    const int n = recurrence * kChunks + chunk_id;

    if (state_history != nullptr) {
      for (int index = threadIdx.x; index < kDim * kValueTile;
           index += blockDim.x) {
        const int key = index / kValueTile;
        const int value = index - key * kValueTile;
        state_history[
            (static_cast<int64_t>(group_n) * kDim + key) * kDim +
            value_start + value] = __float2bfloat16_rn(local_state[index]);
      }
    }
    __syncthreads();

    // Z = U - W H. Four warps own the four 16-row token tiles.
    if (warp < kChunk / kMatrix) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int row_start = warp * kMatrix;
      for (int key_start = 0; key_start < kDim; key_start += kMatrix) {
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const int64_t offset =
              (static_cast<int64_t>(group_n) * kChunk + row_start + row) *
              kDim + key_start + column;
          operand_a[warp * kTileElements + element] =
              __float2bfloat16_rn(W[offset]);
        }
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment, operand_a + warp * kTileElements, kMatrix);
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(local_state[
                  (key_start + row) * kValueTile +
                  column]);
        }
        __syncwarp();
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
      }
      wmma::store_matrix_sync(
          z + row_start * kValueTile,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kValueTile;
         index += blockDim.x) {
      const int row = index / kValueTile;
      const int value = index - row * kValueTile;
      const int64_t offset =
          (static_cast<int64_t>(group_n) * kChunk + row) * kDim +
          value_start + value;
      z[index] = U[offset] - z[index];
      if (z_history != nullptr) {
        z_history[offset] = __float2bfloat16_rn(z[index]);
      }
    }
    __syncthreads();

    // next_state = E^T Z. Eight warps own the eight 16-row key tiles;
    // the 16-value CTA tile removes value-half serialization.
    if (warp < kDim / kMatrix) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int key_start = warp * kMatrix;
      for (int source_start = 0; source_start < kChunk;
           source_start += kMatrix) {
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const int64_t offset =
              (static_cast<int64_t>(group_n) * kChunk +
               source_start + column) * kDim + key_start + row;
          operand_a[warp * kTileElements + element] =
              __float2bfloat16_rn(E[offset]);
        }
        __syncwarp();
        MatrixA a_fragment;
        wmma::load_matrix_sync(
            a_fragment, operand_a + warp * kTileElements, kMatrix);
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(z[
                  (source_start + row) * kValueTile + column]);
        }
        __syncwarp();
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
      }
      wmma::store_matrix_sync(
          next_state + key_start * kValueTile,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      const int key = index / kValueTile;
      local_state[index] =
          expf(prefix_g[chunk_vector_index(n, kChunk - 1, key)]) *
          local_state[index] + next_state[index];
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < kDim * kValueTile;
       index += blockDim.x) {
    const int key = index / kValueTile;
    const int value = index - key * kValueTile;
    state[(static_cast<int64_t>(recurrence) * kDim + key) * kDim +
          value_start + value] = local_state[index];
  }
}

// Rebuild only Z from saved incoming chunk states. Unlike the boundary sweep,
// chunks are independent here: no E^T Z state update or inter-chunk barrier is
// required. The saved BF16 state is expanded once into the FP32 H workspace
// used by the dense VJP products.
__global__ void nanochat_kda_wy_backward_z_from_history_wmma_c64_kernel(
    const __nv_bfloat16* state_history,
    const float* U,
    const float* W,
    float* H,
    float* z_history,
    __nv_bfloat16* z_history_bf16,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kValueTile = 16;
  constexpr int kMatrix = 16;
  constexpr int kTileElements = kMatrix * kMatrix;

  const int owner = blockIdx.x;
  if (owner >= kRecurrences * (kDim / kValueTile)) {
    return;
  }
  const int value_tile = owner % (kDim / kValueTile);
  const int recurrence = owner / (kDim / kValueTile);
  const int value_start = value_tile * kValueTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  __shared__ float z[kChunk * kValueTile];
  __shared__ __nv_bfloat16 operand_a[4 * kTileElements];
  __shared__ __nv_bfloat16 operand_b[4 * kTileElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kMatrix, kMatrix, kMatrix,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kMatrix, kMatrix, kMatrix,
      __nv_bfloat16, wmma::row_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kMatrix, kMatrix, kMatrix, float>;

  for (int local_chunk = 0; local_chunk < group_chunks; ++local_chunk) {
    const int group_n = recurrence * group_chunks + local_chunk;
    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      const int key = index / kValueTile;
      const int value = index - key * kValueTile;
      const int64_t offset =
          (static_cast<int64_t>(group_n) * kDim + key) * kDim +
          value_start + value;
      if (H != nullptr) {
        H[offset] = __bfloat162float(state_history[offset]);
      }
    }
    __syncthreads();

    if (warp < kChunk / kMatrix) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int row_start = warp * kMatrix;
      for (int key_start = 0; key_start < kDim; key_start += kMatrix) {
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const int64_t vector_offset =
              (static_cast<int64_t>(group_n) * kChunk + row_start + row) *
              kDim + key_start + column;
          const int64_t state_offset =
              (static_cast<int64_t>(group_n) * kDim + key_start + row) *
              kDim + value_start + column;
          operand_a[warp * kTileElements + element] =
              __float2bfloat16_rn(W[vector_offset]);
          operand_b[warp * kTileElements + element] =
              state_history[state_offset];
        }
        __syncwarp();
        MatrixA a_fragment;
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            a_fragment, operand_a + warp * kTileElements, kMatrix);
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
      }
      wmma::store_matrix_sync(
          z + row_start * kValueTile,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kValueTile;
         index += blockDim.x) {
      const int row = index / kValueTile;
      const int value = index - row * kValueTile;
      const int64_t offset =
          (static_cast<int64_t>(group_n) * kChunk + row) * kDim +
          value_start + value;
      const float value_to_store = U[offset] - z[index];
      z_history[offset] = value_to_store;
      z_history_bf16[offset] = __float2bfloat16_rn(value_to_store);
    }
    __syncthreads();
  }
}

__global__ void nanochat_kda_wy_backward_fp32_to_bf16_c64_kernel(
    const float* source, __nv_bfloat16* destination, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] = __float2bfloat16_rn(source[index]);
  }
}

__global__ void nanochat_kda_wy_backward_pack_bf16_pqt_c64_kernel(
    const float* P,
    const float* Q,
    const float* T,
    __nv_bfloat16* P_bf16,
    __nv_bfloat16* Q_bf16,
    __nv_bfloat16* T_bf16,
    int vector_elements,
    int matrix_elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < vector_elements) {
    P_bf16[index] = __float2bfloat16_rn(P[index]);
    Q_bf16[index] = __float2bfloat16_rn(Q[index]);
  }
  if (index < matrix_elements) {
    T_bf16[index] = __float2bfloat16_rn(T[index]);
  }
}

// Compute the two group-local WY operands together. Each four-warp CTA owns
// one chunk and one sixteen-column value tile; the FP32 source tiles are
// rounded only at the tensor-core boundary and both products retain FP32
// accumulators and outputs.
__global__ void nanochat_kda_wy_backward_group_uw_wmma_c64_kernel(
    const float* T,
    const float* P,
    const float* Q,
    float* U,
    float* W,
    __nv_bfloat16* P_bf16,
    __nv_bfloat16* Q_bf16,
    __nv_bfloat16* T_bf16,
    int group_rows) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTile = 16;
  constexpr int kTileElements = kTile * kTile;
  constexpr int kValueTiles = kDim / kTile;

  const int owner = blockIdx.x;
  const int local_n = owner / kValueTiles;
  const int value_tile = owner - local_n * kValueTiles;
  if (local_n >= group_rows) {
    return;
  }
  const int value_start = value_tile * kTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int row_start = warp * kTile;

  __shared__ __align__(16) __nv_bfloat16 left[4 * kTileElements];
  __shared__ __align__(16) __nv_bfloat16 right[2 * kTileElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kTile, kTile, kTile, float>;

  Accumulator u_accumulator;
  Accumulator w_accumulator;
  wmma::fill_fragment(u_accumulator, 0.0f);
  wmma::fill_fragment(w_accumulator, 0.0f);

  for (int inner_start = 0; inner_start < kChunk; inner_start += kTile) {
    for (int element = lane; element < kTileElements; element += warpSize) {
      const int row = element / kTile;
      const int inner = element - row * kTile;
      const int64_t source =
          (static_cast<int64_t>(local_n) * kChunk + row_start + row) *
              kChunk + inner_start + inner;
      const __nv_bfloat16 rounded = __float2bfloat16_rn(T[source]);
      left[warp * kTileElements + element] = rounded;
      if (T_bf16 != nullptr && value_tile == 0) {
        T_bf16[source] = rounded;
      }
    }
    for (int element = threadIdx.x; element < kTileElements;
         element += blockDim.x) {
      const int inner = element / kTile;
      const int value = element - inner * kTile;
      const int64_t source =
          (static_cast<int64_t>(local_n) * kChunk + inner_start + inner) *
          kDim + value_start + value;
      const __nv_bfloat16 rounded_p = __float2bfloat16_rn(P[source]);
      const __nv_bfloat16 rounded_q = __float2bfloat16_rn(Q[source]);
      right[element] = rounded_p;
      right[kTileElements + element] = rounded_q;
      if (P_bf16 != nullptr) {
        P_bf16[source] = rounded_p;
        Q_bf16[source] = rounded_q;
      }
    }
    __syncthreads();

    MatrixA t_fragment;
    MatrixB p_fragment;
    MatrixB q_fragment;
    wmma::load_matrix_sync(
        t_fragment, left + warp * kTileElements, kTile);
    wmma::load_matrix_sync(p_fragment, right, kTile);
    wmma::load_matrix_sync(q_fragment, right + kTileElements, kTile);
    wmma::mma_sync(
        u_accumulator, t_fragment, p_fragment, u_accumulator);
    wmma::mma_sync(
        w_accumulator, t_fragment, q_fragment, w_accumulator);
    __syncthreads();
  }

  const int64_t destination =
      (static_cast<int64_t>(local_n) * kChunk + row_start) * kDim +
      value_start;
  wmma::store_matrix_sync(
      U + destination, u_accumulator, kDim, wmma::mem_row_major);
  wmma::store_matrix_sync(
      W + destination, w_accumulator, kDim, wmma::mem_row_major);
}

// Fuse the two reverse-group products that share dO. Eight warps own the
// 128 state-adjoint rows; the first four also own the 64 dZ rows. BF16 tensor
// operands feed FP32 accumulators and both results remain FP32 for the exact
// sequential reverse scan.
__global__ void nanochat_kda_wy_backward_group_reverse_products_wmma_c64_kernel(
    const float* R,
    const float* A,
    const __nv_bfloat16* grad_output,
    float* dstate_base,
    float* dZ,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTile = 16;
  constexpr int kTileElements = kTile * kTile;
  constexpr int kValueTiles = kDim / kTile;

  const int owner = blockIdx.x;
  const int local_n = owner / kValueTiles;
  const int value_tile = owner - local_n * kValueTiles;
  if (local_n >= kRecurrences * group_chunks) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int local_chunk = local_n - recurrence * group_chunks;
  const int chunk_id = chunk_start + local_chunk;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token_start = chunk_id * kChunk;
  const int value_start = value_tile * kTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int output_row_start = warp * kTile;

  __shared__ __align__(16) __nv_bfloat16 state_left[8 * kTileElements];
  __shared__ __align__(16) __nv_bfloat16 dz_left[4 * kTileElements];
  __shared__ __align__(16) __nv_bfloat16 right[kTileElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kTile, kTile, kTile, float>;

  Accumulator state_accumulator;
  Accumulator dz_accumulator;
  wmma::fill_fragment(state_accumulator, 0.0f);
  wmma::fill_fragment(dz_accumulator, 0.0f);

  for (int inner_start = 0; inner_start < kChunk; inner_start += kTile) {
    for (int element = lane; element < kTileElements; element += warpSize) {
      const int output_row = element / kTile;
      const int inner = element - output_row * kTile;
      const int64_t r_source =
          (static_cast<int64_t>(local_n) * kChunk + inner_start + inner) *
          kDim + output_row_start + output_row;
      state_left[warp * kTileElements + element] =
          __float2bfloat16_rn(R[r_source]);
      if (warp < kChunk / kTile) {
        const int64_t a_source =
            (static_cast<int64_t>(local_n) * kChunk + inner_start + inner) *
            kChunk + output_row_start + output_row;
        dz_left[warp * kTileElements + element] =
            __float2bfloat16_rn(A[a_source]);
      }
    }
    for (int element = threadIdx.x; element < kTileElements;
         element += blockDim.x) {
      const int inner = element / kTile;
      const int value = element - inner * kTile;
      right[element] = grad_output[input_vector_index(
          b, token_start + inner_start + inner, h, value_start + value)];
    }
    __syncthreads();

    MatrixA state_fragment;
    MatrixB do_fragment;
    wmma::load_matrix_sync(
        state_fragment, state_left + warp * kTileElements, kTile);
    wmma::load_matrix_sync(do_fragment, right, kTile);
    wmma::mma_sync(
        state_accumulator, state_fragment, do_fragment, state_accumulator);
    if (warp < kChunk / kTile) {
      MatrixA a_fragment;
      wmma::load_matrix_sync(
          a_fragment, dz_left + warp * kTileElements, kTile);
      wmma::mma_sync(
          dz_accumulator, a_fragment, do_fragment, dz_accumulator);
    }
    __syncthreads();
  }

  const int64_t state_destination =
      (static_cast<int64_t>(local_n) * kDim + output_row_start) * kDim +
      value_start;
  wmma::store_matrix_sync(
      dstate_base + state_destination, state_accumulator,
      kDim, wmma::mem_row_major);
  if (warp < kChunk / kTile) {
    const int64_t dz_destination =
        (static_cast<int64_t>(local_n) * kChunk + output_row_start) * kDim +
        value_start;
    wmma::store_matrix_sync(
        dZ + dz_destination, dz_accumulator, kDim, wmma::mem_row_major);
  }
}

// Form dA = dO Z^T directly from the original BF16 gradient and the compact
// BF16 Z history emitted by the forward boundary sweep. Four warps own the
// token-row tiles for one output-column tile; FP32 accumulation is retained.
__global__ void nanochat_kda_wy_backward_group_da_wmma_c64_kernel(
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* z_history,
    float* dA,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTile = 16;
  constexpr int kTileElements = kTile * kTile;
  constexpr int kColumnTiles = kChunk / kTile;

  const int owner = blockIdx.x;
  const int local_n = owner / kColumnTiles;
  const int column_tile = owner - local_n * kColumnTiles;
  if (local_n >= kRecurrences * group_chunks) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int local_chunk = local_n - recurrence * group_chunks;
  const int chunk_id = chunk_start + local_chunk;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token_start = chunk_id * kChunk;
  const int column_start = column_tile * kTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int row_start = warp * kTile;

  __shared__ __align__(16) __nv_bfloat16 left[4 * kTileElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kTile, kTile, kTile,
      __nv_bfloat16, wmma::col_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kTile, kTile, kTile, float>;

  Accumulator accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int value_start = 0; value_start < kDim; value_start += kTile) {
    for (int element = lane; element < kTileElements; element += warpSize) {
      const int row = element / kTile;
      const int value = element - row * kTile;
      left[warp * kTileElements + element] = grad_output[input_vector_index(
          b, token_start + row_start + row, h, value_start + value)];
    }
    __syncwarp();
    MatrixA do_fragment;
    MatrixB z_fragment;
    wmma::load_matrix_sync(
        do_fragment, left + warp * kTileElements, kTile);
    const int64_t z_offset =
        (static_cast<int64_t>(local_n) * kChunk + column_start) * kDim +
        value_start;
    wmma::load_matrix_sync(z_fragment, z_history + z_offset, kDim);
    wmma::mma_sync(accumulator, do_fragment, z_fragment, accumulator);
  }
  const int64_t destination =
      (static_cast<int64_t>(local_n) * kChunk + row_start) * kChunk +
      column_start;
  wmma::store_matrix_sync(
      dA + destination, accumulator, kChunk, wmma::mem_row_major);
}

__global__ void nanochat_kda_wy_backward_sub_c64_kernel(
    float* destination, const float* source, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] -= source[index];
  }
}

// Gather the strided recurrence-major slices needed by one reverse chunk and
// save its incoming state adjoint in the group history with one launch.
__global__ void nanochat_kda_wy_backward_prepare_reverse_chunk_c64_kernel(
    const float* dstate_next,
    const float* dstate_base_group,
    const float* dZ_group,
    float* dstate_next_group,
    float* local_dstate,
    float* local_dZ,
    int local_chunk,
    int group_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kStatePerRecurrence = kDim * kDim;
  constexpr int kVectorPerRecurrence = kChunk * kDim;
  constexpr int kStateElements = kRecurrences * kStatePerRecurrence;
  constexpr int kVectorElements = kRecurrences * kVectorPerRecurrence;
  if (index < kStateElements) {
    const int recurrence = index / kStatePerRecurrence;
    const int within = index - recurrence * kStatePerRecurrence;
    const int group_n = recurrence * group_chunks + local_chunk;
    const int64_t group_offset =
        static_cast<int64_t>(group_n) * kStatePerRecurrence + within;
    dstate_next_group[group_offset] = dstate_next[index];
    local_dstate[index] = dstate_base_group[group_offset];
  }
  if (index < kVectorElements) {
    const int recurrence = index / kVectorPerRecurrence;
    const int within = index - recurrence * kVectorPerRecurrence;
    const int group_n = recurrence * group_chunks + local_chunk;
    local_dZ[index] = dZ_group[
        static_cast<int64_t>(group_n) * kVectorPerRecurrence + within];
  }
}

// Scatter the completed chunk adjoints and advance the recurrent state
// adjoint together, replacing two generic strided copy launches.
__global__ void nanochat_kda_wy_backward_finish_reverse_chunk_c64_kernel(
    const float* local_dstate,
    const float* local_dZ,
    float* dstate_next,
    float* dZ_group,
    int local_chunk,
    int group_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int kStatePerRecurrence = kDim * kDim;
  constexpr int kVectorPerRecurrence = kChunk * kDim;
  constexpr int kStateElements = kRecurrences * kStatePerRecurrence;
  constexpr int kVectorElements = kRecurrences * kVectorPerRecurrence;
  if (index < kStateElements) {
    dstate_next[index] = local_dstate[index];
  }
  if (index < kVectorElements) {
    const int recurrence = index / kVectorPerRecurrence;
    const int within = index - recurrence * kVectorPerRecurrence;
    const int group_n = recurrence * group_chunks + local_chunk;
    dZ_group[static_cast<int64_t>(group_n) * kVectorPerRecurrence + within] =
        local_dZ[index];
  }
}

__global__ void nanochat_kda_wy_backward_negate_c64_kernel(
    float* values, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    values[index] = -values[index];
  }
}

__global__ void nanochat_kda_wy_backward_pack_grad_output_c64_kernel(
    const __nv_bfloat16* grad_output, float* packed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int elements = kBatch * kLength * kHeads * kDim;
  if (index >= elements) {
    return;
  }
  const int value = index % kDim;
  const int h = (index / kDim) % kHeads;
  const int token = (index / (kDim * kHeads)) % kLength;
  const int b = index / (kLength * kHeads * kDim);
  const int chunk_id = token / kChunk;
  const int row = token - chunk_id * kChunk;
  const int recurrence = b * kHeads + h;
  const int n = recurrence * kChunks + chunk_id;
  packed[chunk_vector_index(n, row, value)] =
      __bfloat162float(grad_output[index]);
}

__global__ void nanochat_kda_wy_backward_pack_group_grad_output_c64_kernel(
    const __nv_bfloat16* grad_output,
    float* packed,
    int chunk_start,
    int group_chunks) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int elements = kRecurrences * group_chunks * kChunk * kDim;
  if (index >= elements) {
    return;
  }
  const int local_n = index / (kChunk * kDim);
  const int recurrence = local_n / group_chunks;
  const int local_chunk = local_n % group_chunks;
  const int within = index - local_n * kChunk * kDim;
  const int row = within / kDim;
  const int value = within - row * kDim;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token = (chunk_start + local_chunk) * kChunk + row;
  packed[index] = __bfloat162float(
      grad_output[input_vector_index(b, token, h, value)]);
}

__global__ void nanochat_kda_wy_backward_boundary_terms_c64_kernel(
    const float* prefix_g,
    const float* state,
    const float* dstate_next,
    float* dstate,
    float* dD,
    int chunk_id) {
  const int recurrence_key = blockIdx.x;
  const int value = threadIdx.x;
  if (recurrence_key >= kRecurrences * kDim || value >= kDim) {
    return;
  }
  const int recurrence = recurrence_key / kDim;
  const int key = recurrence_key - recurrence * kDim;
  const int n = recurrence * kChunks + chunk_id;
  const int64_t state_offset =
      (static_cast<int64_t>(recurrence) * kDim + key) * kDim + value;
  __shared__ float D;
  __shared__ float dD_terms[kDim];
  if (value == 0) {
    D = expf(prefix_g[chunk_vector_index(n, kChunk - 1, key)]);
  }
  dD_terms[value] = dstate_next[state_offset] * state[state_offset];
  __syncthreads();
  dstate[state_offset] += D * dstate_next[state_offset];
  if (value == 0) {
    float sum = 0.0f;
    for (int reduction_value = 0; reduction_value < kDim;
         ++reduction_value) {
      sum += dD_terms[reduction_value];
    }
    dD[(static_cast<int64_t>(recurrence) * kChunks + chunk_id) * kDim + key] =
        sum;
  }
}

// One CTA owns sixteen adjacent value columns for one recurrence and carries
// their state adjoint through an entire reverse group. This replaces the two
// dependent FP32 BMMs and transfer/arithmetic launch chain for every chunk.
// WMMA operands are rounded to BF16 while every product accumulates in FP32.
__global__ void nanochat_kda_wy_backward_reverse_group_wmma_c64_kernel(
    const float* prefix_g,
    const float* W,
    const float* E,
    const float* dstate_base,
    const float* dZ_group,
    __nv_bfloat16* dZ_history,
    float* dstate_next,
    __nv_bfloat16* dstate_next_history,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kValueTile = 16;
  constexpr int kMatrix = 16;
  constexpr int kWarps = 8;
  constexpr int kTileElements = kMatrix * kMatrix;

  const int owner = blockIdx.x;
  if (owner >= kRecurrences * (kDim / kValueTile)) {
    return;
  }
  const int value_tile = owner % (kDim / kValueTile);
  const int recurrence = owner / (kDim / kValueTile);
  const int value_start = value_tile * kValueTile;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  __shared__ float local_state[kDim * kValueTile];
  __shared__ float local_dZ[kChunk * kValueTile];
  __shared__ float product_state[kDim * kValueTile];
  __shared__ float decay[kDim];
  __shared__ __nv_bfloat16 operand_a[kWarps * kTileElements];
  __shared__ __nv_bfloat16 operand_b[kWarps * kTileElements];

  for (int index = threadIdx.x; index < kDim * kValueTile;
       index += blockDim.x) {
    const int key = index / kValueTile;
    const int value = index - key * kValueTile;
    local_state[index] = dstate_next[
        (static_cast<int64_t>(recurrence) * kDim + key) * kDim +
        value_start + value];
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

  for (int local_chunk = group_chunks; local_chunk-- > 0;) {
    const int group_n = recurrence * group_chunks + local_chunk;
    const int chunk_id = chunk_start + local_chunk;
    const int n = recurrence * kChunks + chunk_id;

    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      const int key = index / kValueTile;
      const int value = index - key * kValueTile;
      dstate_next_history[
          (static_cast<int64_t>(group_n) * kDim + key) * kDim +
          value_start + value] = __float2bfloat16_rn(local_state[index]);
    }
    __syncthreads();

    // local_dZ = dZ_base + E dstate_next. Four warps own token-row tiles.
    if (warp < kChunk / kMatrix) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int row_start = warp * kMatrix;
      for (int key_start = 0; key_start < kDim; key_start += kMatrix) {
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const int64_t offset =
              (static_cast<int64_t>(group_n) * kChunk + row_start + row) *
              kDim + key_start + column;
          operand_a[warp * kTileElements + element] =
              __float2bfloat16_rn(E[offset]);
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(local_state[
                  (key_start + row) * kValueTile + column]);
        }
        __syncwarp();
        MatrixA a_fragment;
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            a_fragment, operand_a + warp * kTileElements, kMatrix);
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
      }
      wmma::store_matrix_sync(
          local_dZ + row_start * kValueTile,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kChunk * kValueTile;
         index += blockDim.x) {
      const int row = index / kValueTile;
      const int value = index - row * kValueTile;
      const int64_t offset =
          (static_cast<int64_t>(group_n) * kChunk + row) * kDim +
          value_start + value;
      local_dZ[index] += dZ_group[offset];
      dZ_history[offset] = __float2bfloat16_rn(local_dZ[index]);
    }
    if (threadIdx.x < kDim) {
      decay[threadIdx.x] = expf(prefix_g[
          chunk_vector_index(n, kChunk - 1, threadIdx.x)]);
    }
    __syncthreads();

    // Form the non-product state terms while local_dZ is ready for W^T dZ.
    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      const int key = index / kValueTile;
      const int value = index - key * kValueTile;
      const int64_t offset =
          (static_cast<int64_t>(group_n) * kDim + key) * kDim +
          value_start + value;
      local_state[index] =
          dstate_base[offset] + decay[key] * local_state[index];
    }
    __syncthreads();

    // product_state = W^T local_dZ. Eight warps own key-row tiles.
    if (warp < kDim / kMatrix) {
      Accumulator accumulator;
      wmma::fill_fragment(accumulator, 0.0f);
      const int key_start = warp * kMatrix;
      for (int source_start = 0; source_start < kChunk;
           source_start += kMatrix) {
        for (int element = lane; element < kTileElements;
             element += warpSize) {
          const int row = element / kMatrix;
          const int column = element - row * kMatrix;
          const int64_t offset =
              (static_cast<int64_t>(group_n) * kChunk +
               source_start + column) * kDim + key_start + row;
          operand_a[warp * kTileElements + element] =
              __float2bfloat16_rn(W[offset]);
          operand_b[warp * kTileElements + element] =
              __float2bfloat16_rn(local_dZ[
                  (source_start + row) * kValueTile + column]);
        }
        __syncwarp();
        MatrixA a_fragment;
        MatrixB b_fragment;
        wmma::load_matrix_sync(
            a_fragment, operand_a + warp * kTileElements, kMatrix);
        wmma::load_matrix_sync(
            b_fragment, operand_b + warp * kTileElements, kMatrix);
        wmma::mma_sync(
            accumulator, a_fragment, b_fragment, accumulator);
      }
      wmma::store_matrix_sync(
          product_state + key_start * kValueTile,
          accumulator, kValueTile, wmma::mem_row_major);
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kDim * kValueTile;
         index += blockDim.x) {
      local_state[index] -= product_state[index];
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < kDim * kValueTile;
       index += blockDim.x) {
    const int key = index / kValueTile;
    const int value = index - key * kValueTile;
    dstate_next[
        (static_cast<int64_t>(recurrence) * kDim + key) * kDim +
        value_start + value] = local_state[index];
  }
}

// FLA's state-backward ownership keeps a 128x32 dh strip resident while it
// walks chunks in reverse.  Two warps split the eight key tiles; each owns two
// adjacent value tiles for every assigned key tile.  The completed per-token
// dZ and incoming per-chunk dh are the only histories published.  In
// particular, R/E, A^T dO, and R^T dO never become global workspaces.
__global__ void nanochat_kda_wy_backward_register_dh_group_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* W,
    const float* A,
    const __nv_bfloat16* grad_output,
    __nv_bfloat16* dZ_history,
    float* dstate_next,
    __nv_bfloat16* dstate_next_history,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  constexpr int kTile = 16;
  constexpr int kValueStrip = 32;
  constexpr int kTileElements = kTile * kTile;
  constexpr int kKeyTiles = kDim / kTile;
  constexpr int kValueTiles = kValueStrip / kTile;
  constexpr int kWarps = 2;

  const int owner = blockIdx.x;
  if (owner >= kRecurrences * (kDim / kValueStrip)) {
    return;
  }
  const int value_strip = owner % (kDim / kValueStrip);
  const int recurrence = owner / (kDim / kValueStrip);
  const int value_start = value_strip * kValueStrip;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  __shared__ float dh_shared[kDim * kValueStrip];
  __shared__ __align__(16) __nv_bfloat16 dz_shared[kChunk * kValueStrip];
  __shared__ __align__(16) __nv_bfloat16 operand_a[kWarps * kTileElements];
  __shared__ __align__(16) __nv_bfloat16 operand_b[kWarps * kTileElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kTile, kTile, kTile,
      __nv_bfloat16, wmma::row_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kTile, kTile, kTile, float>;

  // Four key tiles per warp times two value tiles.  Keeping these fragments
  // live is the essential difference from the shared-state eight-warp scan.
  Accumulator dh_fragment[4][kValueTiles];
#pragma unroll
  for (int owned_key = 0; owned_key < 4; ++owned_key) {
    const int key_tile = warp + owned_key * kWarps;
#pragma unroll
    for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
      const int64_t source =
          (static_cast<int64_t>(recurrence) * kDim + key_tile * kTile) *
              kDim + value_start + value_tile * kTile;
      wmma::load_matrix_sync(
          dh_fragment[owned_key][value_tile],
          dstate_next + source, kDim, wmma::mem_row_major);
    }
  }

  for (int local_chunk = group_chunks; local_chunk-- > 0;) {
    const int group_n = recurrence * group_chunks + local_chunk;
    const int chunk_id = chunk_start + local_chunk;
    const int n = recurrence * kChunks + chunk_id;
    const int h = recurrence % kHeads;
    const int b = recurrence / kHeads;
    const int token_start = chunk_id * kChunk;

    // Publish the incoming chunk dh and expose a BF16-rounded copy for kg*dh.
#pragma unroll
    for (int owned_key = 0; owned_key < 4; ++owned_key) {
      const int key_tile = warp + owned_key * kWarps;
#pragma unroll
      for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
        wmma::store_matrix_sync(
            dh_shared + key_tile * kTile * kValueStrip + value_tile * kTile,
            dh_fragment[owned_key][value_tile], kValueStrip,
            wmma::mem_row_major);
      }
    }
    __syncthreads();
    for (int index = threadIdx.x; index < kDim * kValueStrip;
         index += blockDim.x) {
      const int key = index / kValueStrip;
      const int value = index - key * kValueStrip;
      dstate_next_history[
          (static_cast<int64_t>(group_n) * kDim + key) * kDim +
          value_start + value] = __float2bfloat16_rn(dh_shared[index]);
    }
    __syncthreads();

    // dZ = A^T dO + kg dh.  Each warp owns alternating token-row tiles and
    // both value tiles; the two products retain independent FP32 sums until
    // the same final elementwise addition used by the prior scan.
    for (int row_tile = warp; row_tile < kChunk / kTile;
         row_tile += kWarps) {
#pragma unroll
      for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
        Accumulator da_do;
        Accumulator kg_dh;
        wmma::fill_fragment(da_do, 0.0f);
        wmma::fill_fragment(kg_dh, 0.0f);
        const int row_start = row_tile * kTile;
        const int local_value_start = value_tile * kTile;

        for (int source_start = 0; source_start < kChunk;
             source_start += kTile) {
          for (int element = lane; element < kTileElements;
               element += warpSize) {
            const int row = element / kTile;
            const int column = element - row * kTile;
            operand_a[warp * kTileElements + element] =
                __float2bfloat16_rn(A[
                    (static_cast<int64_t>(group_n) * kChunk +
                     source_start + column) * kChunk + row_start + row]);
            operand_b[warp * kTileElements + element] = grad_output[
                input_vector_index(
                    b, token_start + source_start + row, h,
                    value_start + local_value_start + column)];
          }
          __syncwarp();
          MatrixA a_operand;
          MatrixB do_operand;
          wmma::load_matrix_sync(
              a_operand, operand_a + warp * kTileElements, kTile);
          wmma::load_matrix_sync(
              do_operand, operand_b + warp * kTileElements, kTile);
          wmma::mma_sync(da_do, a_operand, do_operand, da_do);
        }

        for (int key_start = 0; key_start < kDim; key_start += kTile) {
          for (int element = lane; element < kTileElements;
               element += warpSize) {
            const int row = element / kTile;
            const int column = element - row * kTile;
            const int64_t vector_source = chunk_vector_index(
                n, row_start + row, key_start + column);
            const float end_g = prefix_g[
                chunk_vector_index(n, kChunk - 1, key_start + column)];
            operand_a[warp * kTileElements + element] =
                __float2bfloat16_rn(
                    khat[vector_source] *
                    expf(end_g - prefix_g[vector_source]));
            operand_b[warp * kTileElements + element] =
                __float2bfloat16_rn(
                    dh_shared[(key_start + row) * kValueStrip +
                              local_value_start + column]);
          }
          __syncwarp();
          MatrixA kg_operand;
          MatrixB dh_operand;
          wmma::load_matrix_sync(
              kg_operand, operand_a + warp * kTileElements, kTile);
          wmma::load_matrix_sync(
              dh_operand, operand_b + warp * kTileElements, kTile);
          wmma::mma_sync(kg_dh, kg_operand, dh_operand, kg_dh);
        }

#pragma unroll
        for (int element = 0; element < da_do.num_elements; ++element) {
          const int local_row = (lane >> 2) + ((element & 2) ? 8 : 0);
          const int local_value = ((lane & 3) << 1) + (element & 1) +
              ((element & 4) ? 8 : 0);
          const int row = row_start + local_row;
          const int value = local_value_start + local_value;
          const __nv_bfloat16 rounded = __float2bfloat16_rn(
              da_do.x[element] + kg_dh.x[element]);
          dz_shared[row * kValueStrip + value] = rounded;
          dZ_history[
              (static_cast<int64_t>(group_n) * kChunk + row) * kDim +
              value_start + value] = rounded;
        }
      }
    }
    __syncthreads();

    // dh = qg^T dO + D*dh - W^T dZ.  Every warp updates only the register
    // fragments it owns, so no state store/reload occurs between chunks.
#pragma unroll
    for (int owned_key = 0; owned_key < 4; ++owned_key) {
      const int key_tile = warp + owned_key * kWarps;
      const int key_start = key_tile * kTile;
#pragma unroll
      for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
        Accumulator qg_do;
        Accumulator w_dz;
        wmma::fill_fragment(qg_do, 0.0f);
        wmma::fill_fragment(w_dz, 0.0f);
        const int local_value_start = value_tile * kTile;
        for (int source_start = 0; source_start < kChunk;
             source_start += kTile) {
          for (int element = lane; element < kTileElements;
               element += warpSize) {
            const int row = element / kTile;
            const int column = element - row * kTile;
            const int64_t vector_source = chunk_vector_index(
                n, source_start + column, key_start + row);
            operand_a[warp * kTileElements + element] =
                __float2bfloat16_rn(
                    qbar[vector_source] * expf(prefix_g[vector_source]));
            operand_b[warp * kTileElements + element] = grad_output[
                input_vector_index(
                    b, token_start + source_start + row, h,
                    value_start + local_value_start + column)];
          }
          __syncwarp();
          MatrixA qg_operand;
          MatrixB do_operand;
          wmma::load_matrix_sync(
              qg_operand, operand_a + warp * kTileElements, kTile);
          wmma::load_matrix_sync(
              do_operand, operand_b + warp * kTileElements, kTile);
          wmma::mma_sync(qg_do, qg_operand, do_operand, qg_do);

          for (int element = lane; element < kTileElements;
               element += warpSize) {
            const int row = element / kTile;
            const int column = element - row * kTile;
            operand_a[warp * kTileElements + element] =
                __float2bfloat16_rn(W[
                    (static_cast<int64_t>(group_n) * kChunk +
                     source_start + column) * kDim + key_start + row]);
            operand_b[warp * kTileElements + element] =
                dz_shared[(source_start + row) * kValueStrip +
                          local_value_start + column];
          }
          __syncwarp();
          MatrixA w_operand;
          MatrixB dz_operand;
          wmma::load_matrix_sync(
              w_operand, operand_a + warp * kTileElements, kTile);
          wmma::load_matrix_sync(
              dz_operand, operand_b + warp * kTileElements, kTile);
          wmma::mma_sync(w_dz, w_operand, dz_operand, w_dz);
        }
#pragma unroll
        for (int element = 0; element < qg_do.num_elements; ++element) {
          const int local_key = (lane >> 2) + ((element & 2) ? 8 : 0);
          const float decay = expf(prefix_g[chunk_vector_index(
              n, kChunk - 1, key_start + local_key)]);
          dh_fragment[owned_key][value_tile].x[element] =
              qg_do.x[element] +
              decay * dh_fragment[owned_key][value_tile].x[element] -
              w_dz.x[element];
        }
      }
    }
    __syncthreads();
  }

  // The prior group consumes this exact FP32 state boundary.
#pragma unroll
  for (int owned_key = 0; owned_key < 4; ++owned_key) {
    const int key_tile = warp + owned_key * kWarps;
#pragma unroll
    for (int value_tile = 0; value_tile < kValueTiles; ++value_tile) {
      const int64_t destination =
          (static_cast<int64_t>(recurrence) * kDim + key_tile * kTile) *
              kDim + value_start + value_tile * kTile;
      wmma::store_matrix_sync(
          dstate_next + destination,
          dh_fragment[owned_key][value_tile], kDim,
          wmma::mem_row_major);
    }
  }
}

__global__ void nanochat_kda_wy_backward_expand_state_history_c64_kernel(
    const __nv_bfloat16* source,
    float* destination,
    int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] = __bfloat162float(source[index]);
  }
}

// Reconstruct dD deterministically from the exact incoming-state history
// emitted by the persistent reverse group kernel.
__global__ void nanochat_kda_wy_backward_group_dD_c64_kernel(
    const __nv_bfloat16* state,
    const __nv_bfloat16* dstate_next_group,
    float* dD,
    int chunk_start,
    int group_chunks) {
  const int group_key = blockIdx.x;
  const int value = threadIdx.x;
  if (group_key >= kRecurrences * group_chunks * kDim || value >= kDim) {
    return;
  }
  const int group_n = group_key / kDim;
  const int key = group_key - group_n * kDim;
  const int recurrence = group_n / group_chunks;
  const int local_chunk = group_n - recurrence * group_chunks;
  const int chunk_id = chunk_start + local_chunk;
  const int64_t offset =
      (static_cast<int64_t>(group_n) * kDim + key) * kDim + value;
  __shared__ float terms[kDim];
  terms[value] = __bfloat162float(dstate_next_group[offset]) *
      __bfloat162float(state[offset]);
  __syncthreads();
  if (value == 0) {
    float sum = 0.0f;
    for (int reduction_value = 0; reduction_value < kDim;
         ++reduction_value) {
      sum += terms[reduction_value];
    }
    dD[(static_cast<int64_t>(recurrence) * kChunks + chunk_id) * kDim + key] =
        sum;
  }
}

// One two-warp CTA owns a complete chunk. For each sixteen-key strip it keeps
// two row fragments per warp for all three independent state products, so H
// and dH are consumed directly from their compressed histories without FP32
// expansion or standalone dR/dE GEMMs.
__global__ void nanochat_kda_wy_backward_two_warp_state_products_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* dO,
    const float* z,
    const float* dZ,
    const __nv_bfloat16* state_history,
    const __nv_bfloat16* dstate_history,
    const float* dD,
    float* dW,
    float* dqbar,
    float* dkhat,
    float* dprefix,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  const int local_n = blockIdx.x;
  if (local_n >= kRecurrences * group_chunks) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int local_chunk = local_n - recurrence * group_chunks;
  const int chunk_id = chunk_start + local_chunk;
  const int n = recurrence * kChunks + chunk_id;
  const int warp = threadIdx.x / warpSize;

  constexpr int kStrip = kMatrixTile;
  constexpr int kStripElements = kChunk * kStrip;
  __shared__ __align__(16) __nv_bfloat16 left[3 * kStripElements];
  __shared__ __align__(16) __nv_bfloat16 right[2 * kStrip * kStrip];
  __shared__ float result[3 * kStripElements];

  using MatrixA = wmma::fragment<
      wmma::matrix_a, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::row_major>;
  using MatrixB = wmma::fragment<
      wmma::matrix_b, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::col_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kStrip, kStrip, kStrip, float>;

  for (int key_start = 0; key_start < kDim; key_start += kStrip) {
    Accumulator dR0;
    Accumulator dR1;
    Accumulator dE0;
    Accumulator dE1;
    Accumulator dW0;
    Accumulator dW1;
    wmma::fill_fragment(dR0, 0.0f);
    wmma::fill_fragment(dR1, 0.0f);
    wmma::fill_fragment(dE0, 0.0f);
    wmma::fill_fragment(dE1, 0.0f);
    wmma::fill_fragment(dW0, 0.0f);
    wmma::fill_fragment(dW1, 0.0f);

    for (int value_start = 0; value_start < kDim; value_start += kStrip) {
      for (int index = threadIdx.x; index < kStripElements;
           index += blockDim.x) {
        const int row = index / kStrip;
        const int value = value_start + index % kStrip;
        const int64_t offset =
            (static_cast<int64_t>(local_n) * kChunk + row) * kDim + value;
        left[index] = __float2bfloat16_rn(dO[offset]);
        left[kStripElements + index] = __float2bfloat16_rn(z[offset]);
        left[2 * kStripElements + index] =
            __float2bfloat16_rn(dZ[offset]);
      }
      for (int index = threadIdx.x; index < kStrip * kStrip;
           index += blockDim.x) {
        const int key = key_start + index / kStrip;
        const int value = value_start + index % kStrip;
        const int64_t state_offset =
            (static_cast<int64_t>(local_n) * kDim + key) * kDim + value;
        right[index] = state_history[state_offset];
        right[kStrip * kStrip + index] = dstate_history[state_offset];
      }
      __syncthreads();

      const int row0 = warp * 2 * kStrip;
      const int row1 = row0 + kStrip;
      MatrixA do0;
      MatrixA do1;
      MatrixA z0;
      MatrixA z1;
      MatrixA dz0;
      MatrixA dz1;
      MatrixB h;
      MatrixB dh;
      wmma::load_matrix_sync(do0, left + row0 * kStrip, kStrip);
      wmma::load_matrix_sync(do1, left + row1 * kStrip, kStrip);
      wmma::load_matrix_sync(z0, left + kStripElements + row0 * kStrip, kStrip);
      wmma::load_matrix_sync(z1, left + kStripElements + row1 * kStrip, kStrip);
      wmma::load_matrix_sync(
          dz0, left + 2 * kStripElements + row0 * kStrip, kStrip);
      wmma::load_matrix_sync(
          dz1, left + 2 * kStripElements + row1 * kStrip, kStrip);
      wmma::load_matrix_sync(h, right, kStrip);
      wmma::load_matrix_sync(dh, right + kStrip * kStrip, kStrip);
      wmma::mma_sync(dR0, do0, h, dR0);
      wmma::mma_sync(dR1, do1, h, dR1);
      wmma::mma_sync(dE0, z0, dh, dE0);
      wmma::mma_sync(dE1, z1, dh, dE1);
      wmma::mma_sync(dW0, dz0, h, dW0);
      wmma::mma_sync(dW1, dz1, h, dW1);
      __syncthreads();
    }

    const int row0 = warp * 2 * kStrip;
    const int row1 = row0 + kStrip;
    wmma::store_matrix_sync(
        result + row0 * kStrip, dR0, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + row1 * kStrip, dR1, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + kStripElements + row0 * kStrip,
        dE0, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + kStripElements + row1 * kStrip,
        dE1, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + 2 * kStripElements + row0 * kStrip,
        dW0, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + 2 * kStripElements + row1 * kStrip,
        dW1, kStrip, wmma::mem_row_major);
    __syncthreads();

    for (int index = threadIdx.x; index < kStripElements;
         index += blockDim.x) {
      const int row = index / kStrip;
      const int key = key_start + index % kStrip;
      const int64_t local =
          (static_cast<int64_t>(local_n) * kChunk + row) * kDim + key;
      const int64_t source = chunk_vector_index(n, row, key);
      const float g = prefix_g[source];
      const float end_g = prefix_g[
          chunk_vector_index(n, kChunk - 1, key)];
      const float dR = result[index];
      const float dE = result[kStripElements + index];
      const float R = qbar[source] * expf(g);
      const float E = khat[source] * expf(end_g - g);
      dqbar[local] = dR * expf(g);
      dkhat[local] = dE * expf(end_g - g);
      dprefix[local] = dR * R - dE * E;
      dW[local] = -result[2 * kStripElements + index];
    }
    __syncthreads();

    if (threadIdx.x < kStrip) {
      const int key = key_start + threadIdx.x;
      float end_contribution = dD[
          static_cast<int64_t>(n) * kDim + key] *
          expf(prefix_g[chunk_vector_index(n, kChunk - 1, key)]);
      for (int row = 0; row < kChunk; ++row) {
        const int index = row * kStrip + threadIdx.x;
        const int64_t source = chunk_vector_index(n, row, key);
        const float end_g = prefix_g[
            chunk_vector_index(n, kChunk - 1, key)];
        const float E = khat[source] * expf(end_g - prefix_g[source]);
        end_contribution += result[kStripElements + index] * E;
      }
      const int64_t end_local =
          (static_cast<int64_t>(local_n) * kChunk + kChunk - 1) *
          kDim + key;
      dprefix[end_local] += end_contribution;
    }
    __syncthreads();
  }
}

// FLA's useful scheduling boundary is broader than the state products above:
// the same chunk program retains the local 64x64 adjoint while key
// and value strips are live, consumes dW immediately, applies the inverse VJP,
// and emits dP/dQ-dependent vector gradients without global intermediates.
__global__ void nanochat_kda_wy_backward_complete_four_warp_vjp_c64_kernel(
    const __nv_bfloat16* v,
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    const __nv_bfloat16* P,
    const __nv_bfloat16* Q,
    const __nv_bfloat16* T,
    const __nv_bfloat16* dO,
    const __nv_bfloat16* z,
    const __nv_bfloat16* dZ,
    const __nv_bfloat16* state_history,
    const __nv_bfloat16* dstate_history,
    const float* dD,
    float* dM,
    float* dqbar,
    float* dkhat,
    __nv_bfloat16* dv,
    float* dbeta,
    float* dprefix,
    int chunk_start,
    int group_chunks) {
  namespace wmma = nvcuda::wmma;
  const int local_n = blockIdx.x;
  if (local_n >= kRecurrences * group_chunks) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int local_chunk = local_n - recurrence * group_chunks;
  const int chunk_id = chunk_start + local_chunk;
  const int n = recurrence * kChunks + chunk_id;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  constexpr int kStrip = kMatrixTile;
  constexpr int kStripElements = kChunk * kStrip;
  constexpr int kMatrixElements = kChunk * kChunk;
  __shared__ __align__(16) __nv_bfloat16 left[kMatrixElements];
  __shared__ float result[kMatrixElements];

  using MatrixARow = wmma::fragment<
      wmma::matrix_a, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::row_major>;
  using MatrixACol = wmma::fragment<
      wmma::matrix_a, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::col_major>;
  using MatrixBRow = wmma::fragment<
      wmma::matrix_b, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::row_major>;
  using MatrixBCol = wmma::fragment<
      wmma::matrix_b, kStrip, kStrip, kStrip,
      __nv_bfloat16, wmma::col_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kStrip, kStrip, kStrip, float>;

  const int64_t matrix_base = static_cast<int64_t>(local_n) * kMatrixElements;
  // Match FLA's four-warp K=V=128 decomposition: each warp owns one 16-row
  // tile and all four column tiles of the local adjoint. This halves the live
  // accumulator set per warp while preserving the exact phase boundaries.
  Accumulator local_adjoint[4];
#pragma unroll
  for (int tile = 0; tile < 4; ++tile) {
    wmma::fill_fragment(local_adjoint[tile], 0.0f);
  }
  const int row_start = warp * kStrip;

  // Value strips contribute dZ P^T to the inverse adjoint. They also produce
  // dP = T^T dZ, which is consumed immediately by dv and dbeta.
  for (int value_start = 0; value_start < kDim; value_start += kStrip) {
    MatrixARow dz_fragment;
    wmma::load_matrix_sync(
        dz_fragment,
        dZ + (static_cast<int64_t>(local_n) * kChunk + row_start) *
            kDim + value_start,
        kDim);
#pragma unroll
    for (int column_tile = 0; column_tile < 4; ++column_tile) {
      MatrixBCol p_fragment;
      wmma::load_matrix_sync(
          p_fragment,
          P + (static_cast<int64_t>(local_n) * kChunk +
               column_tile * kStrip) * kDim + value_start,
          kDim);
      wmma::mma_sync(
          local_adjoint[column_tile], dz_fragment, p_fragment,
          local_adjoint[column_tile]);
    }

    Accumulator dP_fragment;
    wmma::fill_fragment(dP_fragment, 0.0f);
    for (int inner_start = 0; inner_start < kChunk;
         inner_start += kStrip) {
      MatrixACol t_fragment;
      MatrixBRow dz_inner_fragment;
      wmma::load_matrix_sync(
          t_fragment, T + matrix_base + inner_start * kChunk + row_start,
          kChunk);
      wmma::load_matrix_sync(
          dz_inner_fragment,
          dZ + (static_cast<int64_t>(local_n) * kChunk + inner_start) *
              kDim + value_start,
          kDim);
      wmma::mma_sync(
          dP_fragment, t_fragment, dz_inner_fragment, dP_fragment);
    }
    // For the fixed SM121 m16n16 accumulator layout, each four-lane subgroup
    // owns two rows and all sixteen columns. Consume dP directly from the
    // fragment instead of round-tripping it through shared memory and two CTA
    // barriers. Each matrix element has exactly one lane/element owner.
    float beta_lower = 0.0f;
    float beta_upper = 0.0f;
#pragma unroll
    for (int element = 0; element < dP_fragment.num_elements; ++element) {
      const int local_row = (lane >> 2) + ((element & 2) ? 8 : 0);
      const int local_value = ((lane & 3) << 1) + (element & 1) +
          ((element & 4) ? 8 : 0);
      const int row = row_start + local_row;
      const int value = value_start + local_value;
      const int token = chunk_id * kChunk + row;
      const int64_t input = input_vector_index(b, token, h, value);
      const float dP_value = dP_fragment.x[element];
      const float beta_row = beta[static_cast<int64_t>(n) * kChunk + row];
      dv[input] = __float2bfloat16_rn(beta_row * dP_value);
      const float contribution =
          dP_value * __bfloat162float(v[input]);
      if (element & 2) {
        beta_upper += contribution;
      } else {
        beta_lower += contribution;
      }
    }
#pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
      beta_lower += __shfl_down_sync(0xffffffffu, beta_lower, offset, 4);
      beta_upper += __shfl_down_sync(0xffffffffu, beta_upper, offset, 4);
    }
    if ((lane & 3) == 0) {
      const int local_row = lane >> 2;
      const int lower_row = row_start + local_row;
      const int upper_row = lower_row + 8;
      const int64_t lower_index =
          static_cast<int64_t>(local_n) * kChunk + lower_row;
      const int64_t upper_index =
          static_cast<int64_t>(local_n) * kChunk + upper_row;
      if (value_start == 0) {
        dbeta[lower_index] = beta_lower;
        dbeta[upper_index] = beta_upper;
      } else {
        dbeta[lower_index] += beta_lower;
        dbeta[upper_index] += beta_upper;
      }
    }
  }

  // Key strips form all boundary-state products. dW is negated into BF16
  // scratch, contributes dW Q^T to the retained adjoint, and feeds
  // dQ = T^T dW without ever reaching global memory.
  for (int key_start = 0; key_start < kDim; key_start += kStrip) {
    Accumulator dR_fragment;
    Accumulator dE_fragment;
    Accumulator dW_fragment;
    wmma::fill_fragment(dR_fragment, 0.0f);
    wmma::fill_fragment(dE_fragment, 0.0f);
    wmma::fill_fragment(dW_fragment, 0.0f);

    for (int value_start = 0; value_start < kDim; value_start += kStrip) {
      MatrixARow do_operand;
      MatrixARow z_operand;
      MatrixARow dz_operand;
      MatrixBCol h_operand;
      MatrixBCol dh_operand;
      wmma::load_matrix_sync(
          do_operand,
          dO + input_vector_index(
              b, chunk_id * kChunk + row_start, h, value_start),
          kHeads * kDim);
      wmma::load_matrix_sync(
          z_operand,
          z + (static_cast<int64_t>(local_n) * kChunk + row_start) *
              kDim + value_start,
          kDim);
      wmma::load_matrix_sync(
          dz_operand,
          dZ + (static_cast<int64_t>(local_n) * kChunk + row_start) *
              kDim + value_start,
          kDim);
      wmma::load_matrix_sync(
          h_operand,
          state_history +
              (static_cast<int64_t>(local_n) * kDim + key_start) *
                  kDim + value_start,
          kDim);
      wmma::load_matrix_sync(
          dh_operand,
          dstate_history +
              (static_cast<int64_t>(local_n) * kDim + key_start) *
                  kDim + value_start,
          kDim);
      wmma::mma_sync(dR_fragment, do_operand, h_operand, dR_fragment);
      wmma::mma_sync(dE_fragment, z_operand, dh_operand, dE_fragment);
      wmma::mma_sync(dW_fragment, dz_operand, h_operand, dW_fragment);
    }

    wmma::store_matrix_sync(
        result + row_start * kStrip, dR_fragment, kStrip,
        wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + kStripElements + row_start * kStrip,
        dE_fragment, kStrip, wmma::mem_row_major);
    wmma::store_matrix_sync(
        result + 2 * kStripElements + row_start * kStrip,
        dW_fragment, kStrip, wmma::mem_row_major);
    __syncthreads();

    for (int index = threadIdx.x; index < kStripElements;
         index += blockDim.x) {
      const int row = index / kStrip;
      const int key = key_start + index % kStrip;
      const int64_t local =
          (static_cast<int64_t>(local_n) * kChunk + row) * kDim + key;
      const int64_t source = chunk_vector_index(n, row, key);
      const float g = prefix_g[source];
      const float end_g =
          prefix_g[chunk_vector_index(n, kChunk - 1, key)];
      const float dR_value = result[index];
      const float dE_value = result[kStripElements + index];
      const float R_value = qbar[source] * expf(g);
      const float E_value = khat[source] * expf(end_g - g);
      dqbar[local] = dR_value * expf(g);
      dkhat[local] = dE_value * expf(end_g - g);
      dprefix[local] =
          dR_value * R_value - dE_value * E_value;
      left[index] = __float2bfloat16_rn(
          -result[2 * kStripElements + index]);
    }
    __syncthreads();

    MatrixARow dw_operand;
    wmma::load_matrix_sync(dw_operand, left + row_start * kStrip, kStrip);
#pragma unroll
    for (int column_tile = 0; column_tile < 4; ++column_tile) {
      MatrixBCol q_operand;
      wmma::load_matrix_sync(
          q_operand,
          Q + (static_cast<int64_t>(local_n) * kChunk +
               column_tile * kStrip) * kDim + key_start,
          kDim);
      wmma::mma_sync(
          local_adjoint[column_tile], dw_operand, q_operand,
          local_adjoint[column_tile]);
    }

    Accumulator dQ_fragment;
    wmma::fill_fragment(dQ_fragment, 0.0f);
    for (int inner_start = 0; inner_start < kChunk;
         inner_start += kStrip) {
      MatrixACol t_fragment;
      MatrixBRow dw_inner_fragment;
      wmma::load_matrix_sync(
          t_fragment, T + matrix_base + inner_start * kChunk + row_start,
          kChunk);
      wmma::load_matrix_sync(
          dw_inner_fragment, left + inner_start * kStrip, kStrip);
      wmma::mma_sync(
          dQ_fragment, t_fragment, dw_inner_fragment, dQ_fragment);
    }

    // Consume the fixed SM121 accumulator layout in registers. As for dP,
    // each four-lane subgroup owns two rows and all sixteen columns. This
    // avoids the dQ shared-memory store/reload and its producer barrier.
    float beta_lower = 0.0f;
    float beta_upper = 0.0f;
#pragma unroll
    for (int element = 0; element < dQ_fragment.num_elements; ++element) {
      const int local_row = (lane >> 2) + ((element & 2) ? 8 : 0);
      const int local_key = ((lane & 3) << 1) + (element & 1) +
          ((element & 4) ? 8 : 0);
      const int row = row_start + local_row;
      const int key = key_start + local_key;
      const int64_t local =
          (static_cast<int64_t>(local_n) * kChunk + row) * kDim + key;
      const int64_t source = chunk_vector_index(n, row, key);
      const float dQ_value = dQ_fragment.x[element];
      const float exp_g = expf(prefix_g[source]);
      const float beta_row = beta[static_cast<int64_t>(n) * kChunk + row];
      dkhat[local] += dQ_value * beta_row * exp_g;
      dprefix[local] += dQ_value * beta_row * exp_g * khat[source];
      const float contribution = dQ_value * exp_g * khat[source];
      if (element & 2) {
        beta_upper += contribution;
      } else {
        beta_lower += contribution;
      }
    }
#pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
      beta_lower += __shfl_down_sync(0xffffffffu, beta_lower, offset, 4);
      beta_upper += __shfl_down_sync(0xffffffffu, beta_upper, offset, 4);
    }
    if ((lane & 3) == 0) {
      const int local_row = lane >> 2;
      const int lower_row = row_start + local_row;
      const int upper_row = lower_row + 8;
      dbeta[static_cast<int64_t>(local_n) * kChunk + lower_row] +=
          beta_lower;
      dbeta[static_cast<int64_t>(local_n) * kChunk + upper_row] +=
          beta_upper;
    }

    // The end-prefix reduction updates row 63 from warp 0, so retain this
    // barrier until every owning warp has completed its direct dQ update.
    __syncthreads();

    if (threadIdx.x < kStrip) {
      const int key = key_start + threadIdx.x;
      float end_contribution =
          dD[static_cast<int64_t>(n) * kDim + key] *
          expf(prefix_g[chunk_vector_index(n, kChunk - 1, key)]);
      for (int row_index = 0; row_index < kChunk; ++row_index) {
        const int index = row_index * kStrip + threadIdx.x;
        const int64_t source = chunk_vector_index(n, row_index, key);
        const float end_g =
            prefix_g[chunk_vector_index(n, kChunk - 1, key)];
        const float E_value =
            khat[source] * expf(end_g - prefix_g[source]);
        end_contribution += result[kStripElements + index] * E_value;
      }
      const int64_t end_local =
          (static_cast<int64_t>(local_n) * kChunk + kChunk - 1) *
          kDim + key;
      dprefix[end_local] += end_contribution;
    }
    __syncthreads();
  }

  // Complete dM = -T^T (dZ P^T + dW Q^T) T^T. The required dM output is
  // used as the deterministic phase boundary between the two products, so no
  // additional matrix workspace or library GEMM is needed.
#pragma unroll
  for (int column_tile = 0; column_tile < 4; ++column_tile) {
    wmma::store_matrix_sync(
        result + row_start * kChunk + column_tile * kStrip,
        local_adjoint[column_tile], kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kMatrixElements;
       index += blockDim.x) {
    left[index] = __float2bfloat16_rn(result[index]);
  }
  __syncthreads();

#pragma unroll
  for (int column_tile = 0; column_tile < 4; ++column_tile) {
    Accumulator transformed;
    wmma::fill_fragment(transformed, 0.0f);
    for (int inner_start = 0; inner_start < kChunk;
         inner_start += kStrip) {
      MatrixACol t_fragment;
      MatrixBRow adjoint_fragment;
      wmma::load_matrix_sync(
          t_fragment, T + matrix_base + inner_start * kChunk + row_start,
          kChunk);
      wmma::load_matrix_sync(
          adjoint_fragment,
          left + inner_start * kChunk + column_tile * kStrip, kChunk);
      wmma::mma_sync(
          transformed, t_fragment, adjoint_fragment, transformed);
    }
    wmma::store_matrix_sync(
        dM + matrix_base + row_start * kChunk + column_tile * kStrip,
        transformed, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kMatrixElements;
       index += blockDim.x) {
    left[index] = __float2bfloat16_rn(dM[matrix_base + index]);
  }
  __syncthreads();

#pragma unroll
  for (int column_tile = 0; column_tile < 4; ++column_tile) {
    Accumulator transformed;
    wmma::fill_fragment(transformed, 0.0f);
    for (int inner_start = 0; inner_start < kChunk;
         inner_start += kStrip) {
      MatrixARow adjoint_fragment;
      MatrixBCol t_fragment;
      wmma::load_matrix_sync(
          adjoint_fragment, left + row_start * kChunk + inner_start,
          kChunk);
      wmma::load_matrix_sync(
          t_fragment,
          T + matrix_base + column_tile * kStrip * kChunk + inner_start,
          kChunk);
      wmma::mma_sync(
          transformed, adjoint_fragment, t_fragment, transformed);
    }
    wmma::store_matrix_sync(
        result + row_start * kChunk + column_tile * kStrip,
        transformed, kChunk, wmma::mem_row_major);
  }
  __syncthreads();
  for (int index = threadIdx.x; index < kMatrixElements;
       index += blockDim.x) {
    dM[matrix_base + index] = -result[index];
  }
}

// Evaluate one conflict-free stable A/M tile pair and apply its complete
// vector VJP in the producer CTA. The color launch is the deterministic
// ordering boundary; shared FP32 WMMA results never round-trip through global
// pair workspaces.
__global__ void nanochat_kda_wy_backward_colored_pair_wmma_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    const float* dA,
    const float* dM,
    float* dqbar,
    float* dkhat,
    float* dbeta,
    float* dprefix,
    int chunk_start,
    int group_chunks,
    int color) {
  namespace wmma = nvcuda::wmma;
  const int pair_n = blockIdx.x;
  const int pair_count = color == 0 ? 4 : 2;
  const int local_n = pair_n / pair_count;
  const int pair_slot = pair_n - local_n * pair_count;
  const int pair = colored_causal_pair(color, pair_slot);
  if (local_n >= kRecurrences * group_chunks) {
    return;
  }
  int target_tile;
  int source_tile;
  causal_tile_pair(pair, target_tile, source_tile);
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  const int target_start = target_tile * kMatrixTile;
  const int source_start = source_tile * kMatrixTile;
  const int center_row = target_tile == source_tile
      ? target_start : target_start - 1;

  __shared__ __nv_bfloat16 shared_upstream[
      2 * kMatrixTile * kMatrixTile];
  __shared__ __nv_bfloat16 shared_right[kMatrixTile * kDim];
  __shared__ __nv_bfloat16 shared_left[2 * kMatrixTile * kDim];
  __shared__ float shared_target_gradient[
      2 * kMatrixTile * kDim];
  __shared__ float shared_source_gradient[kMatrixTile * kDim];
  __shared__ float shared_pre_m[kMatrixTile * kMatrixTile];

  for (int index = threadIdx.x; index < kMatrixTile * kDim;
       index += blockDim.x) {
    const int local_row = index / kDim;
    const int d = index - local_row * kDim;
    const int target_row = target_start + local_row;
    const int source_row = source_start + local_row;
    const int64_t target = chunk_vector_index(n, target_row, d);
    const int64_t source = chunk_vector_index(n, source_row, d);
    const float center = prefix_g[chunk_vector_index(n, center_row, d)];
    const float target_factor = expf(prefix_g[target] - center);
    const float query_left = qbar[target] * target_factor;
    const float key_left = khat[target] * target_factor;
    const float right_value =
        khat[source] * expf(center - prefix_g[source]);
    shared_left[index] = __float2bfloat16_rn(query_left);
    shared_left[kMatrixTile * kDim + index] =
        __float2bfloat16_rn(key_left);
    shared_right[index] = __float2bfloat16_rn(right_value);
  }
  for (int index = threadIdx.x;
       index < 2 * kMatrixTile * kMatrixTile; index += blockDim.x) {
    const int stacked_row = index / kMatrixTile;
    const int source_local = index - stacked_row * kMatrixTile;
    const bool matrix_m = stacked_row >= kMatrixTile;
    const int target_local = matrix_m ? stacked_row - kMatrixTile : stacked_row;
    const int target_row = target_start + target_local;
    const int source_row = source_start + source_local;
    const int64_t source = chunk_matrix_index(local_n, target_row, source_row);
    const float value = matrix_m
        ? (source_row < target_row
            ? dM[source] * beta[static_cast<int64_t>(n) * kChunk + target_row]
            : 0.0f)
        : (source_row <= target_row ? dA[source] : 0.0f);
    shared_upstream[index] = __float2bfloat16_rn(value);
  }
  __syncthreads();

  using MatrixARow = wmma::fragment<
      wmma::matrix_a, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixACol = wmma::fragment<
      wmma::matrix_a, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::col_major>;
  using MatrixBRow = wmma::fragment<
      wmma::matrix_b, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::row_major>;
  using MatrixBCol = wmma::fragment<
      wmma::matrix_b, kMatrixTile, kMatrixTile, kMatrixTile,
      __nv_bfloat16, wmma::col_major>;
  using Accumulator = wmma::fragment<
      wmma::accumulator, kMatrixTile, kMatrixTile, kMatrixTile, float>;
  const int warp = threadIdx.x / warpSize;

  // [32,16] @ [16,128] -> [32,128].
  for (int tile = warp; tile < 2 * (kDim / kMatrixTile);
       tile += blockDim.x / warpSize) {
    const int row_tile = tile / (kDim / kMatrixTile);
    const int value_tile = tile - row_tile * (kDim / kMatrixTile);
    MatrixARow a;
    MatrixBRow b;
    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    wmma::load_matrix_sync(
        a, shared_upstream + row_tile * kMatrixTile * kMatrixTile,
        kMatrixTile);
    wmma::load_matrix_sync(
        b, shared_right + value_tile * kMatrixTile, kDim);
    wmma::mma_sync(accumulator, a, b, accumulator);
    wmma::store_matrix_sync(
        shared_target_gradient +
            row_tile * kMatrixTile * kDim + value_tile * kMatrixTile,
        accumulator, kDim, wmma::mem_row_major);
  }

  // [16,32] @ [32,128] -> [16,128].
  if (warp < kDim / kMatrixTile) {
    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int stacked_start = 0; stacked_start < 2 * kMatrixTile;
         stacked_start += kMatrixTile) {
      MatrixACol a;
      MatrixBRow b;
      wmma::load_matrix_sync(
          a, shared_upstream + stacked_start * kMatrixTile, kMatrixTile);
      wmma::load_matrix_sync(
          b, shared_left + stacked_start * kDim + warp * kMatrixTile,
          kDim);
      wmma::mma_sync(accumulator, a, b, accumulator);
    }
    wmma::store_matrix_sync(
        shared_source_gradient + warp * kMatrixTile,
        accumulator, kDim, wmma::mem_row_major);
  }

  // [16,128] @ [128,16] -> [16,16].
  if (warp == 0) {
    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (int key_start = 0; key_start < kDim; key_start += kMatrixTile) {
      MatrixARow a;
      MatrixBCol b;
      wmma::load_matrix_sync(
          a, shared_left + kMatrixTile * kDim + key_start, kDim);
      wmma::load_matrix_sync(
          b, shared_right + key_start, kDim);
      wmma::mma_sync(accumulator, a, b, accumulator);
    }
    wmma::store_matrix_sync(
        shared_pre_m, accumulator, kMatrixTile,
        wmma::mem_row_major);
  }

  __syncthreads();

  // Each color owns disjoint tiles. Threads apply target and source terms in a
  // fixed order, including the diagonal case where both refer to one tile.
  for (int index = threadIdx.x; index < kMatrixTile * kDim;
       index += blockDim.x) {
    const int local_row = index / kDim;
    const int d = index - local_row * kDim;
    const int target_row = target_start + local_row;
    const int source_row = source_start + local_row;
    const int64_t target = chunk_vector_index(n, target_row, d);
    const int64_t source = chunk_vector_index(n, source_row, d);
    const float center = prefix_g[chunk_vector_index(n, center_row, d)];
    const float target_factor = expf(prefix_g[target] - center);
    const float query_left = qbar[target] * target_factor;
    const float key_left = khat[target] * target_factor;
    const float query_gradient = shared_target_gradient[index];
    const float target_key_gradient =
        shared_target_gradient[kMatrixTile * kDim + index];
    const int64_t target_local =
        chunk_vector_index(local_n, target_row, d);
    dqbar[target_local] += query_gradient * target_factor;
    dkhat[target_local] += target_key_gradient * target_factor;
    dprefix[target_local] +=
        query_gradient * query_left + target_key_gradient * key_left;

    const float source_factor = expf(center - prefix_g[source]);
    const float right_value = khat[source] * source_factor;
    const float source_key_gradient = shared_source_gradient[index];
    const int64_t source_local =
        chunk_vector_index(local_n, source_row, d);
    dkhat[source_local] += source_key_gradient * source_factor;
    dprefix[source_local] -= source_key_gradient * right_value;
  }

  if (threadIdx.x < kMatrixTile) {
    const int target_local_row = threadIdx.x;
    const int target_row = target_start + target_local_row;
    float beta_gradient = 0.0f;
    for (int source_local = 0; source_local < kMatrixTile; ++source_local) {
      const int source_row = source_start + source_local;
      if (source_row < target_row) {
        beta_gradient +=
            dM[chunk_matrix_index(local_n, target_row, source_row)] *
            shared_pre_m[target_local_row * kMatrixTile + source_local];
      }
    }
    dbeta[static_cast<int64_t>(local_n) * kChunk + target_row] +=
        beta_gradient;
  }
}

__global__ void nanochat_kda_wy_backward_accumulate_pair_tiles_c64_kernel(
    const float* prefix_g,
    const float* forward_left,
    const float* right,
    const float* target_gradient,
    const float* source_gradient,
    const float* pre_m,
    const float* dM,
    float* dqbar,
    float* dkhat,
    float* dbeta,
    float* dprefix,
    int chunk_start,
    int group_chunks,
    int pair_start,
    int pair_count) {
  const int local_n = blockIdx.x / kChunk;
  const int row = blockIdx.x - local_n * kChunk;
  const int d = threadIdx.x;
  if (local_n >= kRecurrences * group_chunks || d >= kDim) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  float query_gradient = 0.0f;
  float key_gradient = 0.0f;
  float prefix_gradient = 0.0f;
  float beta_gradient = 0.0f;
  for (int local_pair = 0; local_pair < pair_count; ++local_pair) {
    const int pair = pair_start + local_pair;
    int target_tile;
    int source_tile;
    causal_tile_pair(pair, target_tile, source_tile);
    const int target_start = target_tile * kMatrixTile;
    const int source_start = source_tile * kMatrixTile;
    const int center_row = target_tile == source_tile
        ? target_start : target_start - 1;
    const float center = prefix_g[chunk_vector_index(n, center_row, d)];
    const int pair_n = local_n * pair_count + local_pair;
    const int64_t left_base = static_cast<int64_t>(pair_n) *
        (2 * kMatrixTile) * kDim;
    const int64_t vector_base = static_cast<int64_t>(pair_n) *
        kMatrixTile * kDim;
    if (row >= target_start && row < target_start + kMatrixTile) {
      const int local_row = row - target_start;
      const int64_t q_index = left_base + local_row * kDim + d;
      const int64_t k_index = left_base +
          (kMatrixTile + local_row) * kDim + d;
      const float target_factor = expf(
          prefix_g[chunk_vector_index(n, row, d)] - center);
      query_gradient += target_gradient[q_index] * target_factor;
      key_gradient += target_gradient[k_index] * target_factor;
      prefix_gradient += target_gradient[q_index] * forward_left[q_index] +
          target_gradient[k_index] * forward_left[k_index];
      if (d == 0) {
        const int64_t matrix_base = static_cast<int64_t>(pair_n) *
            kMatrixTile * kMatrixTile;
        for (int source_local = 0; source_local < kMatrixTile;
             ++source_local) {
          const int source_row = source_start + source_local;
          if (source_row < row) {
            beta_gradient +=
                dM[chunk_matrix_index(local_n, row, source_row)] *
                pre_m[matrix_base + local_row * kMatrixTile + source_local];
          }
        }
      }
    }
    if (row >= source_start && row < source_start + kMatrixTile) {
      const int local_row = row - source_start;
      const int64_t source_index = vector_base + local_row * kDim + d;
      const float source_factor = expf(
          center - prefix_g[chunk_vector_index(n, row, d)]);
      key_gradient += source_gradient[source_index] * source_factor;
      prefix_gradient -= source_gradient[source_index] * right[source_index];
    }
  }
  const int64_t local = chunk_vector_index(local_n, row, d);
  dqbar[local] += query_gradient;
  dkhat[local] += key_gradient;
  dprefix[local] += prefix_gradient;
  if (d == 0) {
    dbeta[static_cast<int64_t>(local_n) * kChunk + row] += beta_gradient;
  }
}

// Vector VJP for all chunks. Pair terms are added by the batched stable-tile
// products above, while each lane retains exact ownership of one key channel.
__global__ void nanochat_kda_chunk_backward_kernel(
    const __nv_bfloat16* v,
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    const float* R,
    const float* E,
    const float* dR,
    const float* dA,
    const float* dM,
    const float* dP,
    const float* dQ,
    const float* dE,
    const float* dD,
    float* dqbar,
    float* dkhat,
    __nv_bfloat16* dv,
    float* dbeta,
    float* dprefix,
    int chunk_start,
    int group_chunks) {
  const int local_n = blockIdx.x / kChunk;
  const int row = blockIdx.x - local_n * kChunk;
  const int d = threadIdx.x;
  __shared__ float dbeta_terms[kDim];
  if (local_n >= kRecurrences * group_chunks || d >= kDim) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token_start = chunk_id * kChunk;
  const int64_t local_base = static_cast<int64_t>(local_n) * kChunk * kDim;

  const int64_t offset = chunk_vector_index(n, row, d);
    const int local = local_base + row * kDim + d;
    const float g = prefix_g[offset];
    const float exp_g = expf(g);
    float query_gradient = dR[local] * exp_g;
    float key_gradient = dQ[local] *
        beta[static_cast<int64_t>(n) * kChunk + row] * exp_g;
    const float end_g = prefix_g[
        chunk_vector_index(n, kChunk - 1, d)];
    const float e_factor = expf(end_g - g);
    key_gradient += dE[local] * e_factor;
    float prefix_gradient = dR[local] * R[local] +
        dQ[local] *
            (beta[static_cast<int64_t>(n) * kChunk + row] *
             exp_g * khat[offset]) -
        dE[local] * E[local];

    const float beta_row = beta[static_cast<int64_t>(n) * kChunk + row];
    if (row == kChunk - 1) {
      float end_contribution = dD[static_cast<int64_t>(n) * kDim + d] * exp_g;
      for (int source = 0; source < kChunk; ++source) {
        const int source_local = local_base + source * kDim + d;
        end_contribution += dE[source_local] * E[source_local];
      }
      prefix_gradient += end_contribution;
    }
    dqbar[local] = query_gradient;
    dkhat[local] = key_gradient;
    dprefix[local] = prefix_gradient;

    const int token = token_start + row;
    const int64_t input = input_vector_index(b, token, h, d);
    dv[input] = __float2bfloat16_rn(beta_row * dP[local]);

  float beta_gradient = 0.0f;
  dbeta_terms[d] = dP[local] * __bfloat162float(v[input]);
  __syncthreads();
  if (d == 0) {
    for (int value = 0; value < kDim; ++value) {
      beta_gradient += dbeta_terms[value];
    }
  }
  __syncthreads();

  dbeta_terms[d] = dQ[local] * expf(prefix_g[offset]) * khat[offset];
  __syncthreads();
  if (d == 0) {
    for (int key = 0; key < kDim; ++key) {
      beta_gradient += dbeta_terms[key];
    }
  }
  __syncthreads();

  if (d == 0) {
    dbeta[static_cast<int64_t>(local_n) * kChunk + row] = beta_gradient;
  }
}

__global__ void nanochat_kda_chunk_backward_finish_two_warp_c64_kernel(
    const __nv_bfloat16* v,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    const float* dP,
    const float* dQ,
    float* dkhat,
    __nv_bfloat16* dv,
    float* dbeta,
    float* dprefix,
    int chunk_start,
    int group_chunks) {
  const int local_n = blockIdx.x / kChunk;
  const int row = blockIdx.x - local_n * kChunk;
  const int d = threadIdx.x;
  __shared__ float dbeta_terms[kDim];
  if (local_n >= kRecurrences * group_chunks || d >= kDim) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token = chunk_id * kChunk + row;
  const int64_t local =
      (static_cast<int64_t>(local_n) * kChunk + row) * kDim + d;
  const int64_t source = chunk_vector_index(n, row, d);
  const float beta_row = beta[static_cast<int64_t>(n) * kChunk + row];
  const float exp_g = expf(prefix_g[source]);
  const float dQ_value = dQ[local];
  dkhat[local] += dQ_value * beta_row * exp_g;
  dprefix[local] += dQ_value * beta_row * exp_g * khat[source];

  const int64_t input = input_vector_index(b, token, h, d);
  dv[input] = __float2bfloat16_rn(beta_row * dP[local]);
  dbeta_terms[d] = dP[local] * __bfloat162float(v[input]) +
      dQ_value * exp_g * khat[source];
  __syncthreads();
  if (d == 0) {
    float sum = 0.0f;
    for (int key = 0; key < kDim; ++key) {
      sum += dbeta_terms[key];
    }
    dbeta[static_cast<int64_t>(local_n) * kChunk + row] = sum;
  }
}

__global__ void nanochat_kda_wy_backward_prefix_reverse_c64_kernel(
    float* dprefix, int group_chunks) {
  const int n = blockIdx.x;
  const int d = threadIdx.x;
  if (n >= kRecurrences * group_chunks || d >= kDim) {
    return;
  }
  float running = 0.0f;
  for (int row = kChunk; row-- > 0;) {
    const int64_t offset = chunk_vector_index(n, row, d);
    running += dprefix[offset];
    dprefix[offset] = running;
  }
}

__global__ void nanochat_kda_wy_backward_finalize_c64_kernel(
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* q_inverse,
    const float* k_inverse,
    const float* qbar,
    const float* khat,
    const float* beta,
    const float* dqbar,
    const float* dkhat,
    const float* dbeta,
    const float* ddecay,
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* draw_gate,
    __nv_bfloat16* dbeta_logits,
    float lower_bound,
    float scale,
    int chunk_start,
    int group_chunks) {
  const int local_n = blockIdx.x / kChunk;
  const int row = blockIdx.x - local_n * kChunk;
  const int d = threadIdx.x;
  if (local_n >= kRecurrences * group_chunks || d >= kDim) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int n = recurrence * kChunks + chunk_id;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const int token_start = chunk_id * kChunk;
  __shared__ float q_contribution[kDim];
  __shared__ float k_contribution[kDim];
  __shared__ float q_dot;
  __shared__ float k_dot;

  const float a = expf(A_log[h]);
  const int64_t offset = chunk_vector_index(n, row, d);
  const int64_t local_offset = chunk_vector_index(local_n, row, d);
  const float normalized_q = qbar[offset] / scale;
  q_contribution[d] = dqbar[local_offset] * normalized_q;
  k_contribution[d] = dkhat[local_offset] * khat[offset];
  __syncthreads();
  if (d == 0) {
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int key = 0; key < kDim; ++key) {
      q_sum += q_contribution[key];
      k_sum += k_contribution[key];
    }
    q_dot = q_sum;
    k_dot = k_sum;
  }
  __syncthreads();
  const int token = token_start + row;
  const int64_t input = input_vector_index(b, token, h, d);
  dq[input] = __float2bfloat16_rn(
      scale * q_inverse[static_cast<int64_t>(n) * kChunk + row] *
      (dqbar[local_offset] - normalized_q * q_dot));
  dk[input] = __float2bfloat16_rn(
      k_inverse[static_cast<int64_t>(n) * kChunk + row] *
      (dkhat[local_offset] - khat[offset] * k_dot));

  const float biased_gate = __bfloat162float(raw_gate[input]) +
      dt_bias[h * kDim + d];
  const float activated = wy_sigmoid(a * biased_gate);
  const float raw_gradient = ddecay[local_offset] * lower_bound *
      activated * (1.0f - activated) * a;
  draw_gate[input] = __float2bfloat16_rn(raw_gradient);
  if (d == 0) {
    const int64_t scalar = input_scalar_index(b, token, h);
    const float beta_value = beta[static_cast<int64_t>(n) * kChunk + row];
    dbeta_logits[scalar] = __float2bfloat16_rn(
        dbeta[static_cast<int64_t>(local_n) * kChunk + row] *
        beta_value * (1.0f - beta_value));
  }
}

__global__ void nanochat_kda_wy_backward_parameter_chunks_c64_kernel(
    const __nv_bfloat16* raw_gate,
    const float* A_log,
    const float* dt_bias,
    const float* ddecay,
    float* chunk_partials,
    float lower_bound,
    int chunk_start,
    int group_chunks) {
  const int local_n = blockIdx.x;
  const int key = threadIdx.x;
  if (local_n >= kRecurrences * group_chunks || key >= kDim) {
    return;
  }
  const int recurrence = local_n / group_chunks;
  const int chunk_id = chunk_start + local_n % group_chunks;
  const int h = recurrence % kHeads;
  const int b = recurrence / kHeads;
  const float a = expf(A_log[h]);
  float bias_gradient = 0.0f;
  float A_gradient = 0.0f;
  for (int row = kChunk; row-- > 0;) {
    const int token = chunk_id * kChunk + row;
    const int64_t input = input_vector_index(b, token, h, key);
    const float biased_gate = __bfloat162float(raw_gate[input]) +
        dt_bias[h * kDim + key];
    const float activated = wy_sigmoid(a * biased_gate);
    const float raw_gradient =
        ddecay[chunk_vector_index(local_n, row, key)] * lower_bound *
        activated * (1.0f - activated) * a;
    bias_gradient += raw_gradient;
    A_gradient += raw_gradient * biased_gate;
  }
  const int64_t base = static_cast<int64_t>(local_n) * 2 * kDim + key;
  chunk_partials[base] = bias_gradient;
  chunk_partials[base + kDim] = A_gradient;
}

__global__ void nanochat_kda_wy_backward_parameter_c64_kernel(
    const float* chunk_partials,
    float* dA_partial,
    float* ddt_partial,
    int group_chunks) {
  const int recurrence = blockIdx.x;
  const int key = threadIdx.x;
  if (recurrence >= kRecurrences || key >= kDim) {
    return;
  }
  __shared__ float A_key_gradients[kDim];
  float bias_gradient = ddt_partial[(recurrence * kDim) + key];
  float A_key_gradient = 0.0f;
  for (int local_chunk = group_chunks; local_chunk-- > 0;) {
    const int local_n = recurrence * group_chunks + local_chunk;
    const int64_t base = static_cast<int64_t>(local_n) * 2 * kDim + key;
    bias_gradient += chunk_partials[base];
    A_key_gradient += chunk_partials[base + kDim];
  }
  ddt_partial[(recurrence * kDim) + key] = bias_gradient;
  A_key_gradients[key] = A_key_gradient;
  __syncthreads();
  if (key == 0) {
    float A_gradient = dA_partial[recurrence];
    for (int reduction_key = 0; reduction_key < kDim; ++reduction_key) {
      A_gradient += A_key_gradients[reduction_key];
    }
    dA_partial[recurrence] = A_gradient;
  }
}

__global__ void nanochat_kda_wy_backward_reduce_parameters_c64_kernel(
    const float* dA_partial,
    const float* ddt_partial,
    float* dA_log,
    float* ddt_bias) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kHeads) {
    float sum = 0.0f;
    for (int b = 0; b < kBatch; ++b) {
      sum += dA_partial[b * kHeads + index];
    }
    dA_log[index] = sum;
  }
  if (index < kHeads * kDim) {
    const int h = index / kDim;
    const int key = index - h * kDim;
    float sum = 0.0f;
    for (int b = 0; b < kBatch; ++b) {
      sum += ddt_partial[((b * kHeads + h) * kDim) + key];
    }
    ddt_bias[index] = sum;
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor>
nanochat_kda_chunk_wy_backward_c64(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& grad_output,
    float lower_bound,
    float scale) {
  const at::TensorOptions fp32 = A_log.options();
  at::NoGradGuard no_grad;
  at::NoTF32Guard no_tf32;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  constexpr int kThreads = 256;
  constexpr int kGroupChunks = 8;
  constexpr int kGroups = kChunks / kGroupChunks;
  constexpr int kGroupRows = kRecurrences * kGroupChunks;
  constexpr int kVectorElements = kRecurrences * kChunk * kDim;
  constexpr int kGroupVectorElements = kGroupRows * kChunk * kDim;
  constexpr int kStateElements = kRecurrences * kDim * kDim;
  constexpr int kGroupMatrixElements = kGroupRows * kChunk * kChunk;

  at::Tensor q_inverse = at::empty({kChunkRows, kChunk}, fp32);
  at::Tensor k_inverse = at::empty_like(q_inverse);
  at::Tensor qbar = at::empty({kChunkRows, kChunk, kDim}, fp32);
  at::Tensor khat = at::empty_like(qbar);
  at::Tensor prefix_g = at::empty_like(qbar);
  at::Tensor beta = at::empty({kChunkRows, kChunk}, fp32);
  at::Tensor P = at::empty_like(qbar);
  at::Tensor Q = at::empty_like(qbar);
  at::Tensor M = at::empty({kChunkRows, kChunk, kChunk}, fp32);
  at::Tensor A = at::empty_like(M);
  at::Tensor T = at::empty_like(M);

  nanochat_kda_wy_backward_preprocess_c64_kernel<<<
      kChunkRows, kDim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          raw_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(
          beta_logits.data_ptr<at::BFloat16>()),
      A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
      q_inverse.data_ptr<float>(), k_inverse.data_ptr<float>(),
      qbar.data_ptr<float>(), khat.data_ptr<float>(),
      prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
      P.data_ptr<float>(), Q.data_ptr<float>(), lower_bound, scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kPairCount =
      (kChunk / kMatrixTile) * (kChunk / kMatrixTile + 1) / 2;
  nanochat_kda_wy_backward_build_pair_wmma_c64_kernel<<<
      kPairCount * kChunkRows, kThreads, 0, stream>>>(
      qbar.data_ptr<float>(), khat.data_ptr<float>(),
      prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
      M.data_ptr<float>(), A.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  nanochat_kda_wy_backward_solve_c64_kernel<<<
      kChunkRows, kChunk, 0, stream>>>(M.data_ptr<float>(), T.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  M = at::Tensor();

  const auto pack_vectors = [&](const at::Tensor& tensor, int chunk_start) {
    return tensor.view({kGroups, kGroupRows, kChunk, kDim})
        .select(0, chunk_start / kGroupChunks);
  };
  const auto pack_matrices = [&](const at::Tensor& tensor, int chunk_start) {
    return tensor.view({kGroups, kGroupRows, kChunk, kChunk})
        .select(0, chunk_start / kGroupChunks);
  };
  at::Tensor state = at::zeros({kRecurrences, kDim, kDim}, fp32);
  at::Tensor chunk_state_history = at::empty(
      {kGroups, kGroupRows, kDim, kDim}, q.options());
  at::Tensor chunk_z_history = at::empty(
      {kGroups, kGroupRows, kChunk, kDim}, q.options());

  for (int group_id = 0; group_id < kGroups; ++group_id) {
    const int chunk_start = group_id * kGroupChunks;
    at::Tensor P_group = pack_vectors(P, chunk_start);
    at::Tensor Q_group = pack_vectors(Q, chunk_start);
    at::Tensor T_group = pack_matrices(T, chunk_start);
    at::Tensor U_group = at::empty_like(P_group);
    at::Tensor W_group = at::empty_like(Q_group);
    nanochat_kda_wy_backward_group_uw_wmma_c64_kernel<<<
        kGroupRows * (kDim / 16), 128, 0, stream>>>(
        T_group.data_ptr<float>(), P_group.data_ptr<float>(),
        Q_group.data_ptr<float>(), U_group.data_ptr<float>(),
        W_group.data_ptr<float>(), nullptr, nullptr, nullptr, kGroupRows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor R_group = at::empty_like(P_group);
    at::Tensor E_group = at::empty_like(Q_group);
    nanochat_kda_wy_backward_pack_group_c64_kernel<<<
        (kGroupVectorElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), R_group.data_ptr<float>(),
        E_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nanochat_kda_wy_backward_group_boundary_wmma_c64_kernel<<<
        kRecurrences * (kDim / 16), 256, 0, stream>>>(
        prefix_g.data_ptr<float>(), U_group.data_ptr<float>(),
        W_group.data_ptr<float>(), E_group.data_ptr<float>(),
        state.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            chunk_state_history.select(0, group_id)
                .data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            chunk_z_history.select(0, group_id)
                .data_ptr<at::BFloat16>()),
        chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  state = at::Tensor();

  at::Tensor dq = at::empty_like(q);
  at::Tensor dk = at::empty_like(k);
  at::Tensor dv = at::empty_like(v);
  at::Tensor draw_gate = at::empty_like(raw_gate);
  at::Tensor dbeta_logits = at::empty_like(beta_logits);
  at::Tensor dA_partial = at::zeros({kRecurrences}, fp32);
  at::Tensor ddt_partial = at::zeros({kRecurrences, kDim}, fp32);
  at::Tensor dD = at::empty({kChunkRows, kDim}, fp32);
  at::Tensor dstate_next = at::zeros({kRecurrences, kDim, kDim}, fp32);

  // Process each reverse group through its local VJP before moving to the
  // previous group. The group-local U/W and packed dO operands are therefore
  // built once and reused by both phases, while dstate_next remains the exact
  // sequential boundary between groups.
  for (int group_id = kGroups; group_id-- > 0;) {
    const int chunk_start = group_id * kGroupChunks;
    at::Tensor P_group = pack_vectors(P, chunk_start);
    at::Tensor Q_group = pack_vectors(Q, chunk_start);
    at::Tensor A_group = pack_matrices(A, chunk_start);
    at::Tensor T_group = pack_matrices(T, chunk_start);
    at::Tensor U_group = at::empty_like(P_group);
    at::Tensor W_group = at::empty_like(Q_group);
    at::Tensor P_group_bf16 = at::empty(
        {kGroupRows, kChunk, kDim}, q.options());
    at::Tensor Q_group_bf16 = at::empty_like(P_group_bf16);
    at::Tensor T_group_bf16 = at::empty(
        {kGroupRows, kChunk, kChunk}, q.options());
    nanochat_kda_wy_backward_group_uw_wmma_c64_kernel<<<
        kGroupRows * (kDim / 16), 128, 0, stream>>>(
        T_group.data_ptr<float>(), P_group.data_ptr<float>(),
        Q_group.data_ptr<float>(), U_group.data_ptr<float>(),
        W_group.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            P_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            Q_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            T_group_bf16.data_ptr<at::BFloat16>()),
        kGroupRows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor dZ_group_bf16 = at::empty(
        {kGroupRows, kChunk, kDim}, q.options());
    at::Tensor dstate_next_group_bf16 = at::empty(
        {kGroupRows, kDim, kDim}, q.options());
    nanochat_kda_wy_backward_register_dh_group_c64_kernel<<<
        kRecurrences * (kDim / 32), 64, 0, stream>>>(
        qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), W_group.data_ptr<float>(),
        A_group.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            dZ_group_bf16.data_ptr<at::BFloat16>()),
        dstate_next.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(
            dstate_next_group_bf16.data_ptr<at::BFloat16>()),
        chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor z_group_bf16 = chunk_z_history.select(0, group_id);

    nanochat_kda_wy_backward_group_dD_c64_kernel<<<
        kGroupRows * kDim, kDim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            chunk_state_history.select(0, group_id)
                .data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            dstate_next_group_bf16.data_ptr<at::BFloat16>()),
        dD.data_ptr<float>(),
        chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor dA_group = at::empty_like(A_group);
    at::Tensor dT_group = at::empty_like(T_group);
    nanochat_kda_wy_backward_group_da_wmma_c64_kernel<<<
        kGroupRows * (kChunk / 16), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            z_group_bf16.data_ptr<at::BFloat16>()),
        dA_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor dqbar_group = at::empty_like(P_group);
    at::Tensor dkhat_group = at::empty_like(Q_group);
    at::Tensor dprefix_group = at::empty_like(P_group);
    at::Tensor dbeta_group = at::empty({kGroupRows, kChunk}, fp32);
    nanochat_kda_wy_backward_complete_four_warp_vjp_c64_kernel<<<
        kGroupRows, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            P_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            Q_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            T_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            z_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            dZ_group_bf16.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            chunk_state_history.select(0, group_id)
                .data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            dstate_next_group_bf16.data_ptr<at::BFloat16>()),
        dD.data_ptr<float>(), dT_group.data_ptr<float>(),
        dqbar_group.data_ptr<float>(), dkhat_group.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
        dbeta_group.data_ptr<float>(),
        dprefix_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor parameter_chunk_partials = at::empty(
        {kGroupRows, 2, kDim}, fp32);
    for (int color = 0; color < 4; ++color) {
      const int pair_count = color == 0 ? 4 : 2;
      nanochat_kda_wy_backward_colored_pair_wmma_c64_kernel<<<
          kGroupRows * pair_count, 256, 0, stream>>>(
          qbar.data_ptr<float>(), khat.data_ptr<float>(),
          prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
          dA_group.data_ptr<float>(), dT_group.data_ptr<float>(),
          dqbar_group.data_ptr<float>(), dkhat_group.data_ptr<float>(),
          dbeta_group.data_ptr<float>(), dprefix_group.data_ptr<float>(),
          chunk_start, kGroupChunks, color);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    nanochat_kda_wy_backward_prefix_reverse_c64_kernel<<<
        kGroupRows, kDim, 0, stream>>>(
        dprefix_group.data_ptr<float>(), kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nanochat_kda_wy_backward_finalize_c64_kernel<<<
        kGroupRows * kChunk, kDim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            raw_gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(
            beta_logits.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        q_inverse.data_ptr<float>(), k_inverse.data_ptr<float>(),
        qbar.data_ptr<float>(), khat.data_ptr<float>(), beta.data_ptr<float>(),
        dqbar_group.data_ptr<float>(), dkhat_group.data_ptr<float>(),
        dbeta_group.data_ptr<float>(), dprefix_group.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dq.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            draw_gate.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(
            dbeta_logits.data_ptr<at::BFloat16>()),
        lower_bound, scale, chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nanochat_kda_wy_backward_parameter_chunks_c64_kernel<<<
        kGroupRows, kDim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            raw_gate.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(), dt_bias.data_ptr<float>(),
        dprefix_group.data_ptr<float>(),
        parameter_chunk_partials.data_ptr<float>(), lower_bound,
        chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nanochat_kda_wy_backward_parameter_c64_kernel<<<
        kRecurrences, kDim, 0, stream>>>(
        parameter_chunk_partials.data_ptr<float>(),
        dA_partial.data_ptr<float>(), ddt_partial.data_ptr<float>(),
        kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  P = at::Tensor();
  Q = at::Tensor();
  A = at::Tensor();
  T = at::Tensor();
  chunk_state_history = at::Tensor();
  dD = at::Tensor();
  dstate_next = at::Tensor();
  at::Tensor dA_log = at::empty_like(A_log);
  at::Tensor ddt_bias = at::empty_like(dt_bias);
  nanochat_kda_wy_backward_reduce_parameters_c64_kernel<<<
      1, kDim * kHeads, 0, stream>>>(
      dA_partial.data_ptr<float>(), ddt_partial.data_ptr<float>(),
      dA_log.data_ptr<float>(), ddt_bias.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv, draw_gate, dbeta_logits, dA_log, ddt_bias};
}
