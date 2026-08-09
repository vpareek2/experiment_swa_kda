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

__device__ __forceinline__ void causal_tile_pair(
    int pair, int& target_tile, int& source_tile) {
  target_tile = 0;
  while (pair >= target_tile + 1) {
    pair -= target_tile + 1;
    ++target_tile;
  }
  source_tile = pair;
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
    P[destination] = beta_value * __bfloat162float(v[source]);
    Q[destination] = beta_value * expf(running_g) * normalized_k;
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
            T[chunk_matrix_index(n, inner, column)];
      }
      value = -sum;
    }
    T[chunk_matrix_index(n, row, column)] = value;
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

__global__ void nanochat_kda_wy_backward_sub_c64_kernel(
    float* destination, const float* source, int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    destination[index] -= source[index];
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

// Pack every causal 16x16 A/M tile pair in an eight-chunk group into two
// stacked FP32 products.  A per-channel tile center bounds both exponentials.
__global__ void nanochat_kda_wy_backward_pack_pair_tiles_c64_kernel(
    const float* qbar,
    const float* khat,
    const float* prefix_g,
    const float* beta,
    const float* dA,
    const float* dM,
    float* upstream,
    float* right,
    float* forward_left,
    int chunk_start,
    int group_chunks,
    int pair_start,
    int pair_count) {
  const int pair_n = blockIdx.x;
  const int d = threadIdx.x;
  const int local_n = pair_n / pair_count;
  const int pair = pair_start + pair_n - local_n * pair_count;
  if (local_n >= kRecurrences * group_chunks || d >= kDim) {
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
  const float center = prefix_g[chunk_vector_index(n, center_row, d)];
  const int64_t vector_base = static_cast<int64_t>(pair_n) *
      kMatrixTile * kDim;
  const int64_t left_base = static_cast<int64_t>(pair_n) *
      (2 * kMatrixTile) * kDim;
  const int64_t matrix_base = static_cast<int64_t>(pair_n) *
      (2 * kMatrixTile) * kMatrixTile;
  for (int local_row = 0; local_row < kMatrixTile; ++local_row) {
    const int target_row = target_start + local_row;
    const int source_row = source_start + local_row;
    const int64_t target = chunk_vector_index(n, target_row, d);
    const int64_t source = chunk_vector_index(n, source_row, d);
    const float target_factor = expf(prefix_g[target] - center);
    forward_left[left_base + local_row * kDim + d] =
        qbar[target] * target_factor;
    forward_left[left_base + (kMatrixTile + local_row) * kDim + d] =
        khat[target] * target_factor;
    right[vector_base + local_row * kDim + d] =
        khat[source] * expf(center - prefix_g[source]);
  }
  if (d < kMatrixTile) {
    for (int target_local = 0; target_local < kMatrixTile; ++target_local) {
      const int target_row = target_start + target_local;
      const int source_row = source_start + d;
      const bool a_active = source_row <= target_row;
      const bool m_active = source_row < target_row;
      const int64_t source = chunk_matrix_index(local_n, target_row, source_row);
      upstream[matrix_base + target_local * kMatrixTile + d] =
          a_active ? dA[source] : 0.0f;
      upstream[matrix_base +
          (kMatrixTile + target_local) * kMatrixTile + d] =
          m_active ? dM[source] * beta[
              static_cast<int64_t>(n) * kChunk + target_row] : 0.0f;
    }
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
  constexpr int kPairBatch = 4;
  constexpr int kPairRows = kGroupRows * kPairBatch;
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
  constexpr int kTileElements = kChunkRows * kMatrixTile * kDim;
  at::Tensor P_flat = P.view({-1});
  at::Tensor tile_q_left = P_flat.narrow(0, 0, kTileElements).view(
      {kChunkRows, kMatrixTile, kDim});
  at::Tensor tile_k_left = P_flat.narrow(
      0, kTileElements, kTileElements).view(
      {kChunkRows, kMatrixTile, kDim});
  at::Tensor tile_right = P_flat.narrow(
      0, 2 * kTileElements, kTileElements).view(
      {kChunkRows, kMatrixTile, kDim});
  for (int target_start = 0; target_start < kChunk;
       target_start += kMatrixTile) {
    for (int source_start = 0; source_start <= target_start;
         source_start += kMatrixTile) {
      nanochat_kda_wy_backward_transform_pair_c64_kernel<<<
          (kTileElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(
          qbar.data_ptr<float>(), khat.data_ptr<float>(),
          prefix_g.data_ptr<float>(), tile_q_left.data_ptr<float>(),
          tile_k_left.data_ptr<float>(),
          tile_right.data_ptr<float>(), target_start, source_start);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      at::Tensor A_tile = A.narrow(1, target_start, kMatrixTile)
          .narrow(2, source_start, kMatrixTile);
      at::bmm_out(A_tile, tile_q_left, tile_right.transpose(1, 2));
      at::Tensor M_tile = M.narrow(1, target_start, kMatrixTile)
          .narrow(2, source_start, kMatrixTile);
      at::bmm_out(M_tile, tile_k_left, tile_right.transpose(1, 2));
    }
  }
  constexpr int kAllMatrixElements = kChunkRows * kChunk * kChunk;
  nanochat_kda_wy_backward_finish_m_a_c64_kernel<<<
      (kAllMatrixElements + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      beta.data_ptr<float>(), M.data_ptr<float>(), A.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  constexpr int kAllVectorElements = kChunkRows * kChunk * kDim;
  nanochat_kda_wy_backward_rebuild_p_c64_kernel<<<
      (kAllVectorElements + kThreads - 1) / kThreads,
      kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      beta.data_ptr<float>(), P.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  nanochat_kda_wy_backward_solve_c64_kernel<<<
      kChunkRows, kChunk, 0, stream>>>(M.data_ptr<float>(), T.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  M = at::Tensor();

  const auto pack_vectors = [&](const at::Tensor& tensor, int chunk_start) {
    return tensor.view({kRecurrences, kChunks, kChunk, kDim})
        .narrow(1, chunk_start, kGroupChunks).contiguous()
        .view({kGroupRows, kChunk, kDim});
  };
  const auto pack_matrices = [&](const at::Tensor& tensor, int chunk_start) {
    return tensor.view({kRecurrences, kChunks, kChunk, kChunk})
        .narrow(1, chunk_start, kGroupChunks).contiguous()
        .view({kGroupRows, kChunk, kChunk});
  };
  at::Tensor state = at::zeros({kRecurrences, kDim, kDim}, fp32);
  at::Tensor group_boundaries = at::empty(
      {kRecurrences, kGroups, kDim, kDim}, fp32);
  at::Tensor z = at::empty({kRecurrences, kChunk, kDim}, fp32);
  at::Tensor state_delta = at::empty_like(state);

  for (int group_id = 0; group_id < kGroups; ++group_id) {
    const int chunk_start = group_id * kGroupChunks;
    group_boundaries.select(1, group_id).copy_(state);
    at::Tensor P_group = pack_vectors(P, chunk_start);
    at::Tensor Q_group = pack_vectors(Q, chunk_start);
    at::Tensor T_group = pack_matrices(T, chunk_start);
    at::Tensor U_group = at::empty_like(P_group);
    at::Tensor W_group = at::empty_like(Q_group);
    at::bmm_out(U_group, T_group, P_group);
    at::bmm_out(W_group, T_group, Q_group);
    at::Tensor R_group = at::empty_like(P_group);
    at::Tensor E_group = at::empty_like(Q_group);
    nanochat_kda_wy_backward_pack_group_c64_kernel<<<
        (kGroupVectorElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), R_group.data_ptr<float>(),
        E_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor U_chunks = U_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor W_chunks = W_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor E_chunks = E_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    for (int local_chunk = 0; local_chunk < kGroupChunks; ++local_chunk) {
      const int chunk_id = chunk_start + local_chunk;
      const at::Tensor W_chunk = W_chunks.select(1, local_chunk);
      at::bmm_out(z, W_chunk, state);
      nanochat_kda_wy_backward_sub_group_z_c64_kernel<<<
          (kVectorElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(U_group.data_ptr<float>(), z.data_ptr<float>(),
          z.data_ptr<float>(), local_chunk, kGroupChunks);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      nanochat_kda_wy_backward_decay_state_c64_kernel<<<
          (kStateElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(
          prefix_g.data_ptr<float>(), state.data_ptr<float>(), chunk_id);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      const at::Tensor E_chunk = E_chunks.select(1, local_chunk);
      at::bmm_out(state_delta, E_chunk.transpose(1, 2), z);
      nanochat_kda_wy_backward_add_c64_kernel<<<
          (kStateElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(state.data_ptr<float>(),
          state_delta.data_ptr<float>(), kStateElements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  state = at::Tensor();
  state_delta = at::Tensor();
  z = at::Tensor();

  at::Tensor dq = at::empty_like(q);
  at::Tensor dk = at::empty_like(k);
  at::Tensor dv = at::empty_like(v);
  at::Tensor draw_gate = at::empty_like(raw_gate);
  at::Tensor dbeta_logits = at::empty_like(beta_logits);
  at::Tensor dA_partial = at::zeros({kRecurrences}, fp32);
  at::Tensor ddt_partial = at::zeros({kRecurrences, kDim}, fp32);
  at::Tensor dD = at::empty({kChunkRows, kDim}, fp32);
  at::Tensor dstate_next = at::zeros({kRecurrences, kDim, kDim}, fp32);

  for (int group_id = kGroups; group_id-- > 0;) {
    const int chunk_start = group_id * kGroupChunks;
    at::Tensor P_group = pack_vectors(P, chunk_start);
    at::Tensor Q_group = pack_vectors(Q, chunk_start);
    at::Tensor A_group = pack_matrices(A, chunk_start);
    at::Tensor T_group = pack_matrices(T, chunk_start);
    at::Tensor U_group = at::empty_like(P_group);
    at::Tensor W_group = at::empty_like(Q_group);
    at::bmm_out(U_group, T_group, P_group);
    at::bmm_out(W_group, T_group, Q_group);
    at::Tensor R_group = at::empty_like(P_group);
    at::Tensor E_group = at::empty_like(Q_group);
    nanochat_kda_wy_backward_pack_group_c64_kernel<<<
        (kGroupVectorElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), R_group.data_ptr<float>(),
        E_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor dO_group = at::empty_like(P_group);
    nanochat_kda_wy_backward_pack_group_grad_output_c64_kernel<<<
        (kGroupVectorElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        dO_group.data_ptr<float>(), chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::Tensor U_group_chunks = U_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor W_group_chunks = W_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor E_group_chunks = E_group.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});

    at::Tensor group_state = group_boundaries.select(1, group_id).contiguous();
    at::Tensor H_group_flat = at::empty({kGroupRows, kDim, kDim}, fp32);
    at::Tensor H_group = H_group_flat.view(
        {kRecurrences, kGroupChunks, kDim, kDim});
    at::Tensor z_group_flat = at::empty({kGroupRows, kChunk, kDim}, fp32);
    at::Tensor z_group = z_group_flat.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor local_z = at::empty({kRecurrences, kChunk, kDim}, fp32);
    at::Tensor local_state_delta = at::empty_like(group_state);

    for (int local_chunk = 0; local_chunk < kGroupChunks; ++local_chunk) {
      const int chunk_id = chunk_start + local_chunk;
      H_group.select(1, local_chunk).copy_(group_state);
      const at::Tensor W_chunk = W_group_chunks.select(1, local_chunk);
      at::bmm_out(local_z, W_chunk, group_state);
      nanochat_kda_wy_backward_sub_group_z_c64_kernel<<<
          (kVectorElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(
          U_group.data_ptr<float>(), local_z.data_ptr<float>(),
          local_z.data_ptr<float>(), local_chunk, kGroupChunks);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      z_group.select(1, local_chunk).copy_(local_z);
      nanochat_kda_wy_backward_decay_state_c64_kernel<<<
          (kStateElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(
          prefix_g.data_ptr<float>(), group_state.data_ptr<float>(), chunk_id);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      const at::Tensor E_chunk = E_group_chunks.select(1, local_chunk);
      at::bmm_out(local_state_delta, E_chunk.transpose(1, 2), local_z);
      nanochat_kda_wy_backward_add_c64_kernel<<<
          (kStateElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(group_state.data_ptr<float>(),
          local_state_delta.data_ptr<float>(), kStateElements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    at::Tensor dstate_base = at::empty({kGroupRows, kDim, kDim}, fp32);
    at::Tensor dZ_group_flat = at::empty_like(dO_group);
    at::bmm_out(dstate_base, R_group.transpose(1, 2), dO_group);
    at::bmm_out(dZ_group_flat, A_group.transpose(1, 2), dO_group);
    at::Tensor dstate_base_chunks = dstate_base.view(
        {kRecurrences, kGroupChunks, kDim, kDim});
    at::Tensor dZ_group = dZ_group_flat.view(
        {kRecurrences, kGroupChunks, kChunk, kDim});
    at::Tensor dstate_next_group_flat = at::empty(
        {kGroupRows, kDim, kDim}, fp32);
    at::Tensor dstate_next_group = dstate_next_group_flat.view(
        {kRecurrences, kGroupChunks, kDim, kDim});
    at::Tensor local_dstate = at::empty_like(dstate_next);
    at::Tensor local_dZ = at::empty({kRecurrences, kChunk, kDim}, fp32);
    at::Tensor temp_vector = at::empty_like(local_dZ);
    at::Tensor temp_state = at::empty_like(dstate_next);

    for (int local_chunk = kGroupChunks; local_chunk-- > 0;) {
      const int chunk_id = chunk_start + local_chunk;
      dstate_next_group.select(1, local_chunk).copy_(dstate_next);
      local_dstate.copy_(dstate_base_chunks.select(1, local_chunk));
      local_dZ.copy_(dZ_group.select(1, local_chunk));
      const at::Tensor E_chunk = E_group_chunks.select(1, local_chunk);
      at::bmm_out(temp_vector, E_chunk, dstate_next);
      nanochat_kda_wy_backward_add_c64_kernel<<<
          (kVectorElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(
          local_dZ.data_ptr<float>(), temp_vector.data_ptr<float>(),
          kVectorElements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      dZ_group.select(1, local_chunk).copy_(local_dZ);
      const at::Tensor H = H_group.select(1, local_chunk);
      nanochat_kda_wy_backward_boundary_terms_c64_kernel<<<
          kRecurrences * kDim, kDim, 0, stream>>>(
          prefix_g.data_ptr<float>(), H.data_ptr<float>(),
          dstate_next.data_ptr<float>(), local_dstate.data_ptr<float>(),
          dD.data_ptr<float>(), chunk_id);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      const at::Tensor W_chunk = W_group_chunks.select(1, local_chunk);
      at::bmm_out(temp_state, W_chunk.transpose(1, 2), local_dZ);
      nanochat_kda_wy_backward_sub_c64_kernel<<<
          (kStateElements + kThreads - 1) / kThreads,
          kThreads, 0, stream>>>(local_dstate.data_ptr<float>(),
          temp_state.data_ptr<float>(), kStateElements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      dstate_next.copy_(local_dstate);
    }

    at::Tensor dR_group = at::empty_like(dO_group);
    at::Tensor dA_group = at::empty_like(A_group);
    at::Tensor dE_group = at::empty_like(E_group);
    at::Tensor dW_group = at::empty_like(W_group);
    at::Tensor dT_group = at::empty_like(T_group);
    at::Tensor dP_group = at::empty_like(P_group);
    at::Tensor dQ_group = at::empty_like(Q_group);
    at::Tensor temp_matrix = at::empty_like(T_group);
    at::bmm_out(dR_group, dO_group, H_group_flat.transpose(1, 2));
    at::bmm_out(dA_group, dO_group, z_group_flat.transpose(1, 2));
    at::bmm_out(dE_group, z_group_flat,
                dstate_next_group_flat.transpose(1, 2));
    at::bmm_out(dW_group, dZ_group_flat, H_group_flat.transpose(1, 2));
    nanochat_kda_wy_backward_negate_c64_kernel<<<
        (kGroupVectorElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(
        dW_group.data_ptr<float>(), kGroupVectorElements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::bmm_out(dT_group, dZ_group_flat, P_group.transpose(1, 2));
    at::bmm_out(temp_matrix, dW_group, Q_group.transpose(1, 2));
    nanochat_kda_wy_backward_add_c64_kernel<<<
        (kGroupMatrixElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(dT_group.data_ptr<float>(),
        temp_matrix.data_ptr<float>(), kGroupMatrixElements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    at::bmm_out(dP_group, T_group.transpose(1, 2), dZ_group_flat);
    at::bmm_out(dQ_group, T_group.transpose(1, 2), dW_group);
    at::bmm_out(temp_matrix, T_group.transpose(1, 2), dT_group);
    at::bmm_out(dT_group, temp_matrix, T_group.transpose(1, 2));
    nanochat_kda_wy_backward_negate_c64_kernel<<<
        (kGroupMatrixElements + kThreads - 1) / kThreads,
        kThreads, 0, stream>>>(dT_group.data_ptr<float>(),
        kGroupMatrixElements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor dqbar_group = at::empty_like(P_group);
    at::Tensor dkhat_group = at::empty_like(Q_group);
    at::Tensor dbeta_group = at::empty({kGroupRows, kChunk}, fp32);
    at::Tensor dprefix_group = at::empty_like(P_group);
    at::Tensor parameter_chunk_partials = at::empty(
        {kGroupRows, 2, kDim}, fp32);

    // Reuse dead group buffers for the complete stable A/M pair VJP.  The
    // three products cover target q/k adjoints, combined source-k adjoints,
    // and the unscaled M dot needed by dbeta without increasing allocation.
    constexpr int64_t kPairUpstreamElements =
        static_cast<int64_t>(kPairRows) * 2 * kMatrixTile * kMatrixTile;
    constexpr int64_t kPairVectorElements =
        static_cast<int64_t>(kPairRows) * kMatrixTile * kDim;
    constexpr int64_t kPairLeftElements = 2 * kPairVectorElements;
    constexpr int64_t kPairMatrixElements =
        static_cast<int64_t>(kPairRows) * kMatrixTile * kMatrixTile;
    at::Tensor U_flat = U_group.view({-1});
    at::Tensor pair_upstream = U_flat.narrow(
        0, 0, kPairUpstreamElements).view(
            {kPairRows, 2 * kMatrixTile, kMatrixTile});
    at::Tensor pair_right = Q_group.view({-1}).narrow(
        0, 0, kPairVectorElements).view(
            {kPairRows, kMatrixTile, kDim});
    at::Tensor pair_forward_left = H_group_flat.view({-1}).narrow(
        0, 0, kPairLeftElements).view(
            {kPairRows, 2 * kMatrixTile, kDim});
    at::Tensor pair_target_gradient = dstate_base.view({-1}).narrow(
        0, 0, kPairLeftElements).view(
            {kPairRows, 2 * kMatrixTile, kDim});
    at::Tensor W_flat = W_group.view({-1});
    at::Tensor pair_source_gradient = W_flat.narrow(
        0, 0, kPairVectorElements).view(
            {kPairRows, kMatrixTile, kDim});
    at::Tensor pair_pre_m = R_group.view({-1}).narrow(
        0, 0, kPairMatrixElements).view(
            {kPairRows, kMatrixTile, kMatrixTile});
    nanochat_kda_chunk_backward_kernel<<<
        kGroupRows * kChunk, kDim, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        qbar.data_ptr<float>(), khat.data_ptr<float>(),
        prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
        R_group.data_ptr<float>(), E_group.data_ptr<float>(),
        dR_group.data_ptr<float>(), dA_group.data_ptr<float>(),
        dT_group.data_ptr<float>(), dP_group.data_ptr<float>(),
        dQ_group.data_ptr<float>(), dE_group.data_ptr<float>(),
        dD.data_ptr<float>(), dqbar_group.data_ptr<float>(),
        dkhat_group.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
        dbeta_group.data_ptr<float>(), dprefix_group.data_ptr<float>(),
        chunk_start, kGroupChunks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for (int pair_start = 0; pair_start < kTilePairs;
         pair_start += kPairBatch) {
      const int pair_count =
          pair_start + kPairBatch <= kTilePairs
          ? kPairBatch : kTilePairs - pair_start;
      const int pair_rows = kGroupRows * pair_count;
      const at::Tensor upstream_batch = pair_upstream.narrow(
          0, 0, pair_rows);
      const at::Tensor right_batch = pair_right.narrow(0, 0, pair_rows);
      const at::Tensor forward_left_batch = pair_forward_left.narrow(
          0, 0, pair_rows);
      at::Tensor target_gradient_batch = pair_target_gradient.narrow(
          0, 0, pair_rows);
      at::Tensor source_gradient_batch = pair_source_gradient.narrow(
          0, 0, pair_rows);
      at::Tensor pre_m_batch = pair_pre_m.narrow(0, 0, pair_rows);
      nanochat_kda_wy_backward_pack_pair_tiles_c64_kernel<<<
          pair_rows, kDim, 0, stream>>>(
          qbar.data_ptr<float>(), khat.data_ptr<float>(),
          prefix_g.data_ptr<float>(), beta.data_ptr<float>(),
          dA_group.data_ptr<float>(), dT_group.data_ptr<float>(),
          upstream_batch.data_ptr<float>(), right_batch.data_ptr<float>(),
          forward_left_batch.data_ptr<float>(), chunk_start, kGroupChunks,
          pair_start, pair_count);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      at::bmm_out(target_gradient_batch, upstream_batch, right_batch);
      at::bmm_out(source_gradient_batch, upstream_batch.transpose(1, 2),
                  forward_left_batch);
      at::bmm_out(pre_m_batch,
                  forward_left_batch.narrow(1, kMatrixTile, kMatrixTile),
                  right_batch.transpose(1, 2));
      nanochat_kda_wy_backward_accumulate_pair_tiles_c64_kernel<<<
          kGroupRows * kChunk, kDim, 0, stream>>>(
          prefix_g.data_ptr<float>(), forward_left_batch.data_ptr<float>(),
          right_batch.data_ptr<float>(),
          target_gradient_batch.data_ptr<float>(),
          source_gradient_batch.data_ptr<float>(),
          pre_m_batch.data_ptr<float>(), dT_group.data_ptr<float>(),
          dqbar_group.data_ptr<float>(), dkhat_group.data_ptr<float>(),
          dbeta_group.data_ptr<float>(), dprefix_group.data_ptr<float>(),
          chunk_start, kGroupChunks, pair_start, pair_count);
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
  group_boundaries = at::Tensor();
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
