#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

__device__ __forceinline__ float chunk_sigmoid(float value) {
  if (value >= 0.0f) {
    const float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__device__ __forceinline__ int64_t q_index(
    int64_t b, int64_t t, int64_t h, int64_t k,
    int64_t length, int64_t heads, int64_t key_dim) {
  return (((b * length + t) * heads + h) * key_dim + k);
}

__device__ __forceinline__ int64_t v_index(
    int64_t b, int64_t t, int64_t h, int64_t v,
    int64_t length, int64_t heads, int64_t value_dim) {
  return (((b * length + t) * heads + h) * value_dim + v);
}

__device__ __forceinline__ int64_t state_index(
    int64_t b, int64_t h, int64_t v, int64_t k,
    int64_t heads, int64_t value_dim, int64_t key_dim) {
  return (((b * heads + h) * value_dim + v) * key_dim + k);
}

__device__ __forceinline__ int64_t history_index(
    int64_t b, int64_t h, int64_t t, int64_t v, int64_t k,
    int64_t heads, int64_t length, int64_t value_dim, int64_t key_dim) {
  return (((((b * heads + h) * (length + 1) + t) * value_dim + v) * key_dim) + k);
}

__global__ void nanochat_kda_chunk_preprocess_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* raw_gate,
    const float* A_log,
    const float* dt_bias,
    float* q_square_sum,
    float* k_square_sum,
    float* q_inverse_norm,
    float* k_inverse_norm,
    float* normalized_q,
    float* normalized_k,
    float* gate_sigmoid,
    float* decay_exponential,
    int64_t heads,
    int64_t key_dim,
    int64_t row_count,
    float lower_bound) {
  const int64_t row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  const int64_t h = row % heads;
  const int64_t key_base = row * key_dim;
  __shared__ float q_inverse_norm_shared;
  __shared__ float k_inverse_norm_shared;

  // Lane zero deliberately preserves the recurrence kernels' ascending-key
  // FP32 accumulation order.  Other lanes only begin after both norms exist.
  if (threadIdx.x == 0) {
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float query_value = __bfloat162float(q[key_base + key]);
      const float key_value = __bfloat162float(k[key_base + key]);
      query_sum += query_value * query_value;
      key_sum += key_value * key_value;
    }
    const float query_inverse = rsqrtf(fmaxf(query_sum, 1.0e-24f));
    const float key_inverse = rsqrtf(fmaxf(key_sum, 1.0e-24f));
    q_square_sum[row] = query_sum;
    k_square_sum[row] = key_sum;
    q_inverse_norm[row] = query_inverse;
    k_inverse_norm[row] = key_inverse;
    q_inverse_norm_shared = query_inverse;
    k_inverse_norm_shared = key_inverse;
  }
  __syncthreads();

  const float a = expf(A_log[h]);
  for (int64_t key = threadIdx.x; key < key_dim; key += blockDim.x) {
    const int64_t offset = key_base + key;
    normalized_q[offset] =
        __bfloat162float(q[offset]) * q_inverse_norm_shared;
    normalized_k[offset] =
        __bfloat162float(k[offset]) * k_inverse_norm_shared;
    const float biased_gate =
        __bfloat162float(raw_gate[offset]) + dt_bias[h * key_dim + key];
    const float activated_gate = chunk_sigmoid(a * biased_gate);
    gate_sigmoid[offset] = activated_gate;
    decay_exponential[offset] = expf(lower_bound * activated_gate);
  }
}

__global__ void nanochat_kda_chunk_forward_generic_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    __nv_bfloat16* output,
    float* state,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float lower_bound,
    float scale) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = batch * heads * value_dim;
  if (linear >= count) {
    return;
  }
  const int64_t value = linear % value_dim;
  const int64_t h = (linear / value_dim) % heads;
  const int64_t b = linear / (heads * value_dim);
  const int64_t state_base = state_index(
      b, h, value, 0, heads, value_dim, key_dim);

  for (int64_t key = 0; key < key_dim; ++key) {
    state[state_base + key] = initial_state == nullptr
        ? 0.0f
        : initial_state[state_base + key];
  }

  const float a = expf(A_log[h]);
  for (int64_t token = 0; token < length; ++token) {
    float q_square_sum = 0.0f;
    float k_square_sum = 0.0f;
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    for (int64_t key = 0; key < key_dim; ++key) {
      const float q_value = __bfloat162float(q[q_base + key]);
      const float k_value = __bfloat162float(k[q_base + key]);
      q_square_sum += q_value * q_value;
      k_square_sum += k_value * k_value;
    }
    const float q_inverse_norm = rsqrtf(fmaxf(q_square_sum, 1.0e-24f));
    const float k_inverse_norm = rsqrtf(fmaxf(k_square_sum, 1.0e-24f));

    float prediction = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float gate_input = __bfloat162float(raw_gate[q_base + key]);
      const float decay = lower_bound * chunk_sigmoid(
          a * (gate_input + dt_bias[h * key_dim + key]));
      const float decayed_state = state[state_base + key] * expf(decay);
      state[state_base + key] = decayed_state;
      prediction += __bfloat162float(k[q_base + key]) *
          k_inverse_norm * decayed_state;
    }

    const int64_t value_offset = v_index(
        b, token, h, value, length, heads, value_dim);
    const float residual = __bfloat162float(v[value_offset]) - prediction;
    const int64_t beta_offset = (b * length + token) * heads + h;
    const float beta = chunk_sigmoid(__bfloat162float(beta_logits[beta_offset]));
    float output_value = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float normalized_key = __bfloat162float(k[q_base + key]) * k_inverse_norm;
      const float updated_state = state[state_base + key] +
          beta * normalized_key * residual;
      state[state_base + key] = updated_state;
      const float normalized_query = __bfloat162float(q[q_base + key]) *
          q_inverse_norm * scale;
      output_value += normalized_query * updated_state;
    }
    output[value_offset] = __float2bfloat16_rn(output_value);
  }
}

__global__ void nanochat_kda_chunk_forward_generic_kernel_shared_cache(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    __nv_bfloat16* output,
    float* state,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float lower_bound,
    float scale) {
  const int64_t block = blockIdx.x;
  if (block >= batch * heads) {
    return;
  }
  const int64_t b = block / heads;
  const int64_t h = block - b * heads;
  const int64_t lane = threadIdx.x;
  const int64_t value = lane;
  const int64_t state_base = state_index(
      b, h, value, 0, heads, value_dim, key_dim);
  extern __shared__ float token_cache[];
  float* normalized_q = token_cache;
  float* normalized_k = normalized_q + key_dim;
  float* decay_exp = normalized_k + key_dim;
  __shared__ float q_inverse_norm_shared;
  __shared__ float k_inverse_norm_shared;
  __shared__ float beta_shared;

  for (int64_t key = 0; key < key_dim; ++key) {
    state[state_base + key] = initial_state == nullptr
        ? 0.0f
        : initial_state[state_base + key];
  }

  const float a = expf(A_log[h]);
  for (int64_t token = 0; token < length; ++token) {
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    if (lane == 0) {
      float q_square_sum = 0.0f;
      float k_square_sum = 0.0f;
      for (int64_t key = 0; key < key_dim; ++key) {
        const float q_value = __bfloat162float(q[q_base + key]);
        const float k_value = __bfloat162float(k[q_base + key]);
        q_square_sum += q_value * q_value;
        k_square_sum += k_value * k_value;
      }
      q_inverse_norm_shared = rsqrtf(fmaxf(q_square_sum, 1.0e-24f));
      k_inverse_norm_shared = rsqrtf(fmaxf(k_square_sum, 1.0e-24f));
      const int64_t beta_offset = (b * length + token) * heads + h;
      beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[beta_offset]));
    }
    __syncthreads();

    for (int64_t key = lane; key < key_dim; key += blockDim.x) {
      normalized_q[key] = __bfloat162float(q[q_base + key]) *
          q_inverse_norm_shared * scale;
      normalized_k[key] = __bfloat162float(k[q_base + key]) *
          k_inverse_norm_shared;
      const float gate_input = __bfloat162float(raw_gate[q_base + key]);
      const float decay = lower_bound * chunk_sigmoid(
          a * (gate_input + dt_bias[h * key_dim + key]));
      decay_exp[key] = expf(decay);
    }
    __syncthreads();

    float prediction = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float decayed_state = state[state_base + key] * decay_exp[key];
      state[state_base + key] = decayed_state;
      prediction += normalized_k[key] * decayed_state;
    }

    const int64_t value_offset = v_index(
        b, token, h, value, length, heads, value_dim);
    const float residual = __bfloat162float(v[value_offset]) - prediction;
    float output_value = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float updated_state = state[state_base + key] +
          beta_shared * normalized_k[key] * residual;
      state[state_base + key] = updated_state;
      output_value += normalized_q[key] * updated_state;
    }
    output[value_offset] = __float2bfloat16_rn(output_value);
    __syncthreads();
  }
}

__global__ void nanochat_kda_chunk_forward_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    __nv_bfloat16* output,
    float* state,
    int64_t batch,
    int64_t length,
    int64_t heads,
    float lower_bound,
    float scale) {
  constexpr int64_t key_dim = 128;
  constexpr int64_t value_dim = 128;
  const int64_t block = blockIdx.x;
  if (block >= batch * heads * value_dim) {
    return;
  }
  const int64_t value = block % value_dim;
  const int64_t recurrence = block / value_dim;
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;
  const int64_t key = threadIdx.x;
  __shared__ float q_square_contribution[128];
  __shared__ float k_square_contribution[128];
  __shared__ float reduction[128];
  __shared__ float q_inverse_norm_shared;
  __shared__ float k_inverse_norm_shared;
  __shared__ float beta_shared;
  __shared__ float residual_shared;

  const int64_t state_offset = state_index(
      b, h, value, key, heads, value_dim, key_dim);
  float current_state = initial_state == nullptr
      ? 0.0f
      : initial_state[state_offset];
  const float a = expf(A_log[h]);

  for (int64_t token = 0; token < length; ++token) {
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    // One lane owns one key, so all three activation loads are coalesced.
    const float query_value = __bfloat162float(q[q_base + key]);
    const float key_value = __bfloat162float(k[q_base + key]);
    const float gate_input = __bfloat162float(raw_gate[q_base + key]);
    q_square_contribution[key] = query_value * query_value;
    k_square_contribution[key] = key_value * key_value;
    __syncthreads();

    if (key == 0) {
      float q_square_sum = 0.0f;
      float k_square_sum = 0.0f;
      // Preserve the parent's ascending-key FP32 norm reductions exactly.
      for (int64_t reduction_key = 0;
           reduction_key < key_dim;
           ++reduction_key) {
        q_square_sum += q_square_contribution[reduction_key];
        k_square_sum += k_square_contribution[reduction_key];
      }
      q_inverse_norm_shared = rsqrtf(fmaxf(q_square_sum, 1.0e-24f));
      k_inverse_norm_shared = rsqrtf(fmaxf(k_square_sum, 1.0e-24f));
      const int64_t beta_offset = (b * length + token) * heads + h;
      beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[beta_offset]));
    }
    __syncthreads();

    // Parenthesization deliberately matches the production forward recurrence:
    // normalized_k * (state * decay), not (normalized_k * state) * decay.
    const float normalized_query =
        (query_value * q_inverse_norm_shared) * scale;
    const float normalized_key = key_value * k_inverse_norm_shared;
    const float decay = lower_bound * chunk_sigmoid(
        a * (gate_input + dt_bias[h * key_dim + key]));
    const float decayed_state = current_state * expf(decay);
    reduction[key] = normalized_key * decayed_state;
    __syncthreads();

    if (key == 0) {
      float prediction = 0.0f;
      for (int64_t reduction_key = 0;
           reduction_key < key_dim;
           ++reduction_key) {
        prediction += reduction[reduction_key];
      }
      const int64_t value_offset = v_index(
          b, token, h, value, length, heads, value_dim);
      residual_shared = __bfloat162float(v[value_offset]) - prediction;
    }
    __syncthreads();

    current_state = decayed_state +
        (beta_shared * normalized_key) * residual_shared;
    reduction[key] = normalized_query * current_state;
    __syncthreads();

    if (key == 0) {
      float output_value = 0.0f;
      for (int64_t reduction_key = 0;
           reduction_key < key_dim;
           ++reduction_key) {
        output_value += reduction[reduction_key];
      }
      const int64_t value_offset = v_index(
          b, token, h, value, length, heads, value_dim);
      output[value_offset] = __float2bfloat16_rn(output_value);
    }
    // No lane may reuse a shared contribution slot for the next token early.
    __syncthreads();
  }

  // The output state is written only after the complete token recurrence.
  state[state_offset] = current_state;
}

__global__ void nanochat_kda_chunk_history_128_kernel(
    const __nv_bfloat16* v,
    const __nv_bfloat16* beta_logits,
    const float* normalized_k,
    const float* decay_exponential,
    const float* initial_state,
    float* state_history,
    float* residual_history,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t value_dim) {
  const int64_t block = blockIdx.x;
  if (block >= batch * heads * value_dim) {
    return;
  }
  const int64_t value = block % value_dim;
  const int64_t recurrence = block / value_dim;
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;
  const int64_t key = threadIdx.x;
  constexpr int64_t key_dim = 128;
  __shared__ float contribution[128];
  __shared__ float residual_shared;
  __shared__ float beta_shared;

  const int64_t state = state_index(
      b, h, value, key, heads, value_dim, key_dim);
  float current_state = initial_state == nullptr ? 0.0f : initial_state[state];
  state_history[history_index(
      b, h, 0, value, key, heads, length, value_dim, key_dim)] = current_state;
  __syncthreads();

  for (int64_t token = 0; token < length; ++token) {
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    const float normalized_key = normalized_k[q_base + key];
    const float decay_factor = decay_exponential[q_base + key];
    const float decayed_state = current_state * decay_factor;
    contribution[key] = (normalized_key * current_state) * decay_factor;
    __syncthreads();

    if (key == 0) {
      float prediction = 0.0f;
      for (int64_t reduction_key = 0; reduction_key < key_dim; ++reduction_key) {
        prediction += contribution[reduction_key];
      }
      const int64_t value_offset = v_index(
          b, token, h, value, length, heads, value_dim);
      residual_shared = __bfloat162float(v[value_offset]) - prediction;
      residual_history[value_offset] = residual_shared;
      const int64_t beta_offset = (b * length + token) * heads + h;
      beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[beta_offset]));
    }
    __syncthreads();

    current_state = decayed_state +
        beta_shared * normalized_key * residual_shared;
    state_history[history_index(
        b, h, token + 1, value, key,
        heads, length, value_dim, key_dim)] = current_state;
    __syncthreads();
  }
}

__global__ void nanochat_kda_chunk_backward_generic_kernel(
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* q_square_sum,
    const float* k_square_sum,
    const float* q_inverse_norm,
    const float* k_inverse_norm,
    const float* normalized_q,
    const float* normalized_k,
    const float* gate_sigmoid,
    const float* decay_exponential,
    const float* initial_state,
    const __nv_bfloat16* grad_output,
    const float* grad_final_state,
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* dv,
    __nv_bfloat16* draw_gate,
    __nv_bfloat16* dbeta_logits,
    float* dA_log_partial,
    float* ddt_bias_partial,
    float* dinitial_state,
    float* state_history,
    float* residual_history,
    float* state_adjoint,
    float* normalized_adjoint,
    float* residual_adjoint,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float lower_bound,
    float scale) {
  const int64_t block = blockIdx.x;
  if (block >= batch * heads) {
    return;
  }
  const int64_t b = block / heads;
  const int64_t h = block - b * heads;
  const int64_t lane = threadIdx.x;
  const int64_t lanes = blockDim.x;
  extern __shared__ float reduction[];
  __shared__ float beta_shared;
  __shared__ float query_dot_shared;
  __shared__ float key_dot_shared;
  __shared__ float batch_A_gradient_shared;

  if (lane == 0) {
    batch_A_gradient_shared = 0.0f;
  }
  const int64_t parameter_base = (b * heads + h) * key_dim;
  for (int64_t key = lane; key < key_dim; key += lanes) {
    ddt_bias_partial[parameter_base + key] = 0.0f;
  }
  __syncthreads();

  const float a = expf(A_log[h]);
  // Each block owns one independent (batch, head) recurrence.
  {
    const int64_t matrix_count = value_dim * key_dim;
    for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
      const int64_t value = linear / key_dim;
      const int64_t key = linear - value * key_dim;
      const int64_t history = history_index(
          b, h, 0, value, key, heads, length, value_dim, key_dim);
      const int64_t state = state_index(
          b, h, value, key, heads, value_dim, key_dim);
      state_history[history] = initial_state == nullptr
          ? 0.0f
          : initial_state[state];
    }
    __syncthreads();

    // The recurrence remains sequential in time, while independent value rows
    // are recomputed concurrently.  Each row keeps the original key reduction
    // order and therefore the V-first FP32 state trajectory.
    for (int64_t token = 0; token < length; ++token) {
      const int64_t q_base = q_index(
          b, token, h, 0, length, heads, key_dim);
      if (lane == 0) {
        const int64_t beta_offset = (b * length + token) * heads + h;
        beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[beta_offset]));
      }
      __syncthreads();

      for (int64_t value = lane; value < value_dim; value += lanes) {
        float prediction = 0.0f;
        for (int64_t key = 0; key < key_dim; ++key) {
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          prediction += normalized_k[q_base + key] * previous *
              decay_exponential[q_base + key];
        }
        const int64_t value_offset = v_index(
            b, token, h, value, length, heads, value_dim);
        const float residual = __bfloat162float(v[value_offset]) - prediction;
        residual_history[value_offset] = residual;
        for (int64_t key = 0; key < key_dim; ++key) {
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          state_history[history_index(
              b, h, token + 1, value, key, heads, length, value_dim, key_dim)] =
              previous * decay_exponential[q_base + key] +
              beta_shared * normalized_k[q_base + key] * residual;
        }
      }
      __syncthreads();
    }

    for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
      const int64_t value = linear / key_dim;
      const int64_t key = linear - value * key_dim;
      const int64_t state = state_index(
          b, h, value, key, heads, value_dim, key_dim);
      state_adjoint[state] = grad_final_state == nullptr
          ? 0.0f
          : grad_final_state[state];
    }
    __syncthreads();

    for (int64_t token = length; token-- > 0;) {
      const int64_t q_base = q_index(
          b, token, h, 0, length, heads, key_dim);
      const int64_t row = (b * length + token) * heads + h;
      if (lane == 0) {
        beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[row]));
      }
      __syncthreads();

      for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
        const int64_t value = linear / key_dim;
        const int64_t key = linear - value * key_dim;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        const int64_t value_offset = v_index(
            b, token, h, value, length, heads, value_dim);
        const float normalized_query = normalized_q[q_base + key] * scale;
        state_adjoint[state] +=
            __bfloat162float(grad_output[value_offset]) * normalized_query;
      }
      __syncthreads();

      for (int64_t key = lane; key < key_dim; key += lanes) {
        float normalized_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t value_offset = v_index(
              b, token, h, value, length, heads, value_dim);
          const int64_t history = history_index(
              b, h, token + 1, value, key, heads, length, value_dim, key_dim);
          normalized_gradient += __bfloat162float(grad_output[value_offset]) *
              state_history[history];
        }
        normalized_adjoint[q_base + key] = normalized_gradient;
      }
      __syncthreads();
      if (lane == 0) {
        float query_dot = 0.0f;
        for (int64_t key = 0; key < key_dim; ++key) {
          query_dot += normalized_adjoint[q_base + key] *
              normalized_q[q_base + key];
        }
        query_dot_shared = query_dot;
      }
      __syncthreads();
      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float projection = q_square_sum[row] > 1.0e-24f
            ? normalized_q[q_base + key] * query_dot_shared
            : 0.0f;
        const float input_gradient = scale * q_inverse_norm[row] *
            (normalized_adjoint[q_base + key] - projection);
        dq[q_base + key] = __float2bfloat16_rn(input_gradient);
      }

      for (int64_t value = lane; value < value_dim; value += lanes) {
        float gradient = 0.0f;
        float beta_partial = 0.0f;
        const float residual = residual_history[v_index(
            b, token, h, value, length, heads, value_dim)];
        for (int64_t key = 0; key < key_dim; ++key) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          gradient += state_adjoint[state] * beta_shared *
              normalized_k[q_base + key];
          beta_partial += state_adjoint[state] *
              normalized_k[q_base + key] * residual;
        }
        residual_adjoint[(b * heads + h) * value_dim + value] = gradient;
        dv[v_index(b, token, h, value, length, heads, value_dim)] =
            __float2bfloat16_rn(gradient);
        reduction[value] = beta_partial;
      }
      __syncthreads();
      if (lane == 0) {
        float beta_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          beta_gradient += reduction[value];
        }
        const int64_t beta_offset = (b * length + token) * heads + h;
        dbeta_logits[beta_offset] = __float2bfloat16_rn(
            beta_gradient * beta_shared * (1.0f - beta_shared));
      }

      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float decay_factor = decay_exponential[q_base + key];
        float normalized_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          const float residual = residual_history[v_index(
              b, token, h, value, length, heads, value_dim)];
          const float residual_gradient = residual_adjoint[
              (b * heads + h) * value_dim + value];
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          normalized_gradient += state_adjoint[state] * beta_shared * residual -
              residual_gradient * previous * decay_factor;
        }
        normalized_adjoint[q_base + key] = normalized_gradient;
      }
      __syncthreads();
      if (lane == 0) {
        float key_dot = 0.0f;
        for (int64_t key = 0; key < key_dim; ++key) {
          key_dot += normalized_adjoint[q_base + key] *
              normalized_k[q_base + key];
        }
        key_dot_shared = key_dot;
      }
      __syncthreads();
      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float projection = k_square_sum[row] > 1.0e-24f
            ? normalized_k[q_base + key] * key_dot_shared
            : 0.0f;
        const float input_gradient = k_inverse_norm[row] *
            (normalized_adjoint[q_base + key] - projection);
        dk[q_base + key] = __float2bfloat16_rn(input_gradient);
      }

      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float gate_input = __bfloat162float(raw_gate[q_base + key]);
        const float biased_gate = gate_input + dt_bias[h * key_dim + key];
        const float activated_gate = gate_sigmoid[q_base + key];
        const float decay_factor = decay_exponential[q_base + key];
        float decay_gradient = 0.0f;
        const float normalized_key = normalized_k[q_base + key];
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          const float residual_gradient = residual_adjoint[
              (b * heads + h) * value_dim + value];
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          const float decayed_state = previous * decay_factor;
          const float decayed_gradient = state_adjoint[state] -
              residual_gradient * normalized_key;
          decay_gradient += decayed_gradient * decayed_state;
          state_adjoint[state] = decayed_gradient * decay_factor;
        }
        const float activated_gradient = decay_gradient * lower_bound *
            activated_gate * (1.0f - activated_gate);
        const float raw_gradient = activated_gradient * a;
        draw_gate[q_base + key] = __float2bfloat16_rn(raw_gradient);
        ddt_bias_partial[parameter_base + key] += raw_gradient;
        reduction[key] = raw_gradient * biased_gate;
      }
      __syncthreads();
      if (lane == 0) {
        for (int64_t key = 0; key < key_dim; ++key) {
          batch_A_gradient_shared += reduction[key];
        }
      }
      __syncthreads();
    }

    if (dinitial_state != nullptr) {
      for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
        const int64_t value = linear / key_dim;
        const int64_t key = linear - value * key_dim;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        dinitial_state[state] = state_adjoint[state];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    dA_log_partial[b * heads + h] = batch_A_gradient_shared;
  }
}

__global__ void nanochat_kda_chunk_backward_128_fallback_kernel(
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* q_square_sum,
    const float* k_square_sum,
    const float* q_inverse_norm,
    const float* k_inverse_norm,
    const float* normalized_q,
    const float* normalized_k,
    const float* gate_sigmoid,
    const float* decay_exponential,
    const float* initial_state,
    const __nv_bfloat16* grad_output,
    const float* grad_final_state,
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* dv,
    __nv_bfloat16* draw_gate,
    __nv_bfloat16* dbeta_logits,
    float* dA_log_partial,
    float* ddt_bias_partial,
    float* dinitial_state,
    float* state_history,
    float* residual_history,
    float* state_adjoint,
    float* normalized_adjoint,
    float* residual_adjoint,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float lower_bound,
    float scale) {
  const int64_t block = blockIdx.x;
  if (block >= batch * heads) {
    return;
  }
  const int64_t b = block / heads;
  const int64_t h = block - b * heads;
  const int64_t lane = threadIdx.x;
  const int64_t lanes = blockDim.x;
  extern __shared__ float reduction[];
  __shared__ float beta_shared;
  __shared__ float query_dot_shared;
  __shared__ float key_dot_shared;
  __shared__ float batch_A_gradient_shared;

  if (lane == 0) {
    batch_A_gradient_shared = 0.0f;
  }
  const int64_t parameter_base = (b * heads + h) * key_dim;
  for (int64_t key = lane; key < key_dim; key += lanes) {
    ddt_bias_partial[parameter_base + key] = 0.0f;
  }
  __syncthreads();

  const float a = expf(A_log[h]);
  // Each block owns one independent (batch, head) recurrence.
  {
    const int64_t matrix_count = value_dim * key_dim;
    // History and residual replay is produced by the row-parallel kernel.
    for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
      const int64_t value = linear / key_dim;
      const int64_t key = linear - value * key_dim;
      const int64_t state = state_index(
          b, h, value, key, heads, value_dim, key_dim);
      state_adjoint[state] = grad_final_state == nullptr
          ? 0.0f
          : grad_final_state[state];
    }
    __syncthreads();

    for (int64_t token = length; token-- > 0;) {
      const int64_t q_base = q_index(
          b, token, h, 0, length, heads, key_dim);
      const int64_t row = (b * length + token) * heads + h;
      if (lane == 0) {
        beta_shared = chunk_sigmoid(__bfloat162float(beta_logits[row]));
      }
      __syncthreads();

      for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
        const int64_t value = linear / key_dim;
        const int64_t key = linear - value * key_dim;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        const int64_t value_offset = v_index(
            b, token, h, value, length, heads, value_dim);
        const float normalized_query = normalized_q[q_base + key] * scale;
        state_adjoint[state] +=
            __bfloat162float(grad_output[value_offset]) * normalized_query;
      }
      __syncthreads();

      for (int64_t key = lane; key < key_dim; key += lanes) {
        float normalized_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t value_offset = v_index(
              b, token, h, value, length, heads, value_dim);
          const int64_t history = history_index(
              b, h, token + 1, value, key, heads, length, value_dim, key_dim);
          normalized_gradient += __bfloat162float(grad_output[value_offset]) *
              state_history[history];
        }
        normalized_adjoint[q_base + key] = normalized_gradient;
      }
      __syncthreads();
      if (lane == 0) {
        float query_dot = 0.0f;
        for (int64_t key = 0; key < key_dim; ++key) {
          query_dot += normalized_adjoint[q_base + key] *
              normalized_q[q_base + key];
        }
        query_dot_shared = query_dot;
      }
      __syncthreads();
      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float projection = q_square_sum[row] > 1.0e-24f
            ? normalized_q[q_base + key] * query_dot_shared
            : 0.0f;
        const float input_gradient = scale * q_inverse_norm[row] *
            (normalized_adjoint[q_base + key] - projection);
        dq[q_base + key] = __float2bfloat16_rn(input_gradient);
      }

      for (int64_t value = lane; value < value_dim; value += lanes) {
        float gradient = 0.0f;
        float beta_partial = 0.0f;
        const float residual = residual_history[v_index(
            b, token, h, value, length, heads, value_dim)];
        for (int64_t key = 0; key < key_dim; ++key) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          gradient += state_adjoint[state] * beta_shared *
              normalized_k[q_base + key];
          beta_partial += state_adjoint[state] *
              normalized_k[q_base + key] * residual;
        }
        residual_adjoint[(b * heads + h) * value_dim + value] = gradient;
        dv[v_index(b, token, h, value, length, heads, value_dim)] =
            __float2bfloat16_rn(gradient);
        reduction[value] = beta_partial;
      }
      __syncthreads();
      if (lane == 0) {
        float beta_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          beta_gradient += reduction[value];
        }
        const int64_t beta_offset = (b * length + token) * heads + h;
        dbeta_logits[beta_offset] = __float2bfloat16_rn(
            beta_gradient * beta_shared * (1.0f - beta_shared));
      }

      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float decay_factor = decay_exponential[q_base + key];
        float normalized_gradient = 0.0f;
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          const float residual = residual_history[v_index(
              b, token, h, value, length, heads, value_dim)];
          const float residual_gradient = residual_adjoint[
              (b * heads + h) * value_dim + value];
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          normalized_gradient += state_adjoint[state] * beta_shared * residual -
              residual_gradient * previous * decay_factor;
        }
        normalized_adjoint[q_base + key] = normalized_gradient;
      }
      __syncthreads();
      if (lane == 0) {
        float key_dot = 0.0f;
        for (int64_t key = 0; key < key_dim; ++key) {
          key_dot += normalized_adjoint[q_base + key] *
              normalized_k[q_base + key];
        }
        key_dot_shared = key_dot;
      }
      __syncthreads();
      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float projection = k_square_sum[row] > 1.0e-24f
            ? normalized_k[q_base + key] * key_dot_shared
            : 0.0f;
        const float input_gradient = k_inverse_norm[row] *
            (normalized_adjoint[q_base + key] - projection);
        dk[q_base + key] = __float2bfloat16_rn(input_gradient);
      }

      for (int64_t key = lane; key < key_dim; key += lanes) {
        const float gate_input = __bfloat162float(raw_gate[q_base + key]);
        const float biased_gate = gate_input + dt_bias[h * key_dim + key];
        const float activated_gate = gate_sigmoid[q_base + key];
        const float decay_factor = decay_exponential[q_base + key];
        float decay_gradient = 0.0f;
        const float normalized_key = normalized_k[q_base + key];
        for (int64_t value = 0; value < value_dim; ++value) {
          const int64_t state = state_index(
              b, h, value, key, heads, value_dim, key_dim);
          const float residual_gradient = residual_adjoint[
              (b * heads + h) * value_dim + value];
          const float previous = state_history[history_index(
              b, h, token, value, key, heads, length, value_dim, key_dim)];
          const float decayed_state = previous * decay_factor;
          const float decayed_gradient = state_adjoint[state] -
              residual_gradient * normalized_key;
          decay_gradient += decayed_gradient * decayed_state;
          state_adjoint[state] = decayed_gradient * decay_factor;
        }
        const float activated_gradient = decay_gradient * lower_bound *
            activated_gate * (1.0f - activated_gate);
        const float raw_gradient = activated_gradient * a;
        draw_gate[q_base + key] = __float2bfloat16_rn(raw_gradient);
        ddt_bias_partial[parameter_base + key] += raw_gradient;
        reduction[key] = raw_gradient * biased_gate;
      }
      __syncthreads();
      if (lane == 0) {
        for (int64_t key = 0; key < key_dim; ++key) {
          batch_A_gradient_shared += reduction[key];
        }
      }
      __syncthreads();
    }

    if (dinitial_state != nullptr) {
      for (int64_t linear = lane; linear < matrix_count; linear += lanes) {
        const int64_t value = linear / key_dim;
        const int64_t key = linear - value * key_dim;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        dinitial_state[state] = state_adjoint[state];
      }
    }
    __syncthreads();
  }
  if (lane == 0) {
    dA_log_partial[b * heads + h] = batch_A_gradient_shared;
  }
}

// The active K=V=128 reverse is split into fixed 64-token chunks and eight
// contiguous 16-value tiles.  Keeping the audited symbol on this tile kernel
// makes the production bottleneck visible to the protected profiler.
__global__ void nanochat_kda_chunk_backward_kernel(
    const __nv_bfloat16* beta_logits,
    const float* normalized_q,
    const float* normalized_k,
    const float* decay_exponential,
    const __nv_bfloat16* grad_output,
    __nv_bfloat16* dv,
    const float* state_history,
    const float* residual_history,
    float* state_adjoint,
    float* residual_adjoint,
    float* query_tile_partial,
    float* key_tile_partial,
    float* decay_tile_partial,
    float* beta_tile_partial,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t chunk_start,
    int64_t chunk_end,
    float scale) {
  constexpr int64_t key_dim = 128;
  constexpr int64_t value_dim = 128;
  constexpr int64_t tile_count = 8;
  constexpr int64_t tile_values = 16;
  constexpr int64_t chunk_capacity = 32;
  const int64_t tile = blockIdx.x % tile_count;
  const int64_t recurrence = blockIdx.x / tile_count;
  if (recurrence >= batch * heads) {
    return;
  }
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;
  const int64_t value_start = tile * tile_values;
  const int64_t lane = threadIdx.x;
  __shared__ float beta_shared;
  __shared__ float beta_value_partial[tile_values];

  for (int64_t token = chunk_end; token-- > chunk_start;) {
    const int64_t local_token = token - chunk_start;
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    if (lane == 0) {
      const int64_t beta_offset = (b * length + token) * heads + h;
      beta_shared = chunk_sigmoid(
          __bfloat162float(beta_logits[beta_offset]));
    }
    __syncthreads();

    // Parent phase 1: output-query contribution to this tile's persistent
    // state-adjoint rows.  The two iterations cover 16 * 128 entries.
    for (int64_t linear = lane;
         linear < tile_values * key_dim;
         linear += blockDim.x) {
      const int64_t local_value = linear / key_dim;
      const int64_t key = linear - local_value * key_dim;
      const int64_t value = value_start + local_value;
      const int64_t state = state_index(
          b, h, value, key, heads, value_dim, key_dim);
      const int64_t value_offset = v_index(
          b, token, h, value, length, heads, value_dim);
      state_adjoint[state] += __bfloat162float(grad_output[value_offset]) *
          (normalized_q[q_base + key] * scale);
    }
    __syncthreads();

    // Parent phase 2: a deterministic ascending-value query partial.
    if (lane < key_dim) {
      const int64_t key = lane;
      float partial = 0.0f;
      for (int64_t local_value = 0;
           local_value < tile_values;
           ++local_value) {
        const int64_t value = value_start + local_value;
        const int64_t value_offset = v_index(
            b, token, h, value, length, heads, value_dim);
        const int64_t history = history_index(
            b, h, token + 1, value, key,
            heads, length, value_dim, key_dim);
        partial += __bfloat162float(grad_output[value_offset]) *
            state_history[history];
      }
      const int64_t partial_offset =
          (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
             key_dim) + key);
      query_tile_partial[partial_offset] = partial;
    }
    __syncthreads();

    // Parent phase 3: each value is unique to this CTA, so dv and the
    // residual adjoint need no cross-tile communication or atomics.
    if (lane < tile_values) {
      const int64_t value = value_start + lane;
      float gradient = 0.0f;
      float beta_partial = 0.0f;
      const float residual = residual_history[v_index(
          b, token, h, value, length, heads, value_dim)];
      for (int64_t key = 0; key < key_dim; ++key) {
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        gradient += state_adjoint[state] * beta_shared *
            normalized_k[q_base + key];
        beta_partial += state_adjoint[state] *
            normalized_k[q_base + key] * residual;
      }
      residual_adjoint[(b * heads + h) * value_dim + value] = gradient;
      dv[v_index(b, token, h, value, length, heads, value_dim)] =
          __float2bfloat16_rn(gradient);
      beta_value_partial[lane] = beta_partial;
    }
    __syncthreads();
    if (lane == 0) {
      float partial = 0.0f;
      for (int64_t local_value = 0;
           local_value < tile_values;
           ++local_value) {
        partial += beta_value_partial[local_value];
      }
      const int64_t partial_offset =
          (((tile * batch + b) * chunk_capacity + local_token) * heads + h);
      beta_tile_partial[partial_offset] = partial;
    }
    __syncthreads();

    // Parent phase 4: normalized-key gradient, before the state adjoint is
    // advanced to the preceding token.
    if (lane < key_dim) {
      const int64_t key = lane;
      const float decay_factor = decay_exponential[q_base + key];
      float partial = 0.0f;
      for (int64_t local_value = 0;
           local_value < tile_values;
           ++local_value) {
        const int64_t value = value_start + local_value;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        const float residual = residual_history[v_index(
            b, token, h, value, length, heads, value_dim)];
        const float residual_gradient = residual_adjoint[
            (b * heads + h) * value_dim + value];
        const float previous = state_history[history_index(
            b, h, token, value, key,
            heads, length, value_dim, key_dim)];
        partial += state_adjoint[state] * beta_shared * residual -
            residual_gradient * previous * decay_factor;
      }
      const int64_t partial_offset =
          (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
             key_dim) + key);
      key_tile_partial[partial_offset] = partial;
    }
    __syncthreads();

    // Parent phase 5: decay gradient and the only update of the owned state
    // adjoints.  Values remain in ascending contiguous order inside the tile.
    if (lane < key_dim) {
      const int64_t key = lane;
      const float decay_factor = decay_exponential[q_base + key];
      const float normalized_key = normalized_k[q_base + key];
      float partial = 0.0f;
      for (int64_t local_value = 0;
           local_value < tile_values;
           ++local_value) {
        const int64_t value = value_start + local_value;
        const int64_t state = state_index(
            b, h, value, key, heads, value_dim, key_dim);
        const float residual_gradient = residual_adjoint[
            (b * heads + h) * value_dim + value];
        const float previous = state_history[history_index(
            b, h, token, value, key,
            heads, length, value_dim, key_dim)];
        const float decayed_state = previous * decay_factor;
        const float decayed_gradient = state_adjoint[state] -
            residual_gradient * normalized_key;
        partial += decayed_gradient * decayed_state;
        state_adjoint[state] = decayed_gradient * decay_factor;
      }
      const int64_t partial_offset =
          (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
             key_dim) + key);
      decay_tile_partial[partial_offset] = partial;
    }
    // No lane may enter the preceding token while another lane still owns
    // an adjoint update for the current token.
    __syncthreads();
  }
}

__global__ void nanochat_kda_chunk_backward_row_finalize_128_kernel(
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* q_square_sum,
    const float* k_square_sum,
    const float* q_inverse_norm,
    const float* k_inverse_norm,
    const float* normalized_q,
    const float* normalized_k,
    const float* gate_sigmoid,
    const float* query_tile_partial,
    const float* key_tile_partial,
    const float* decay_tile_partial,
    const float* beta_tile_partial,
    __nv_bfloat16* dq,
    __nv_bfloat16* dk,
    __nv_bfloat16* draw_gate,
    __nv_bfloat16* dbeta_logits,
    float* raw_gate_gradient_history,
    int64_t batch,
    int64_t length,
    int64_t heads,
    int64_t chunk_start,
    int64_t chunk_length,
    float lower_bound,
    float scale) {
  constexpr int64_t key_dim = 128;
  constexpr int64_t tile_count = 8;
  constexpr int64_t chunk_capacity = 32;
  const int64_t local_token = blockIdx.x % chunk_length;
  const int64_t recurrence = blockIdx.x / chunk_length;
  if (recurrence >= batch * heads) {
    return;
  }
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;
  const int64_t token = chunk_start + local_token;
  const int64_t row = (b * length + token) * heads + h;
  const int64_t q_base = q_index(
      b, token, h, 0, length, heads, key_dim);
  const int64_t key = threadIdx.x;
  __shared__ float normalized_gradient[128];
  __shared__ float dot_shared;

  float partial = 0.0f;
  for (int64_t tile = 0; tile < tile_count; ++tile) {
    const int64_t partial_offset =
        (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
           key_dim) + key);
    partial += query_tile_partial[partial_offset];
  }
  normalized_gradient[key] = partial;
  __syncthreads();
  if (key == 0) {
    float dot = 0.0f;
    for (int64_t reduction_key = 0;
         reduction_key < key_dim;
         ++reduction_key) {
      dot += normalized_gradient[reduction_key] *
          normalized_q[q_base + reduction_key];
    }
    dot_shared = dot;
  }
  __syncthreads();
  const float query_projection = q_square_sum[row] > 1.0e-24f
      ? normalized_q[q_base + key] * dot_shared
      : 0.0f;
  dq[q_base + key] = __float2bfloat16_rn(
      scale * q_inverse_norm[row] *
      (normalized_gradient[key] - query_projection));
  __syncthreads();

  partial = 0.0f;
  for (int64_t tile = 0; tile < tile_count; ++tile) {
    const int64_t partial_offset =
        (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
           key_dim) + key);
    partial += key_tile_partial[partial_offset];
  }
  normalized_gradient[key] = partial;
  __syncthreads();
  if (key == 0) {
    float dot = 0.0f;
    for (int64_t reduction_key = 0;
         reduction_key < key_dim;
         ++reduction_key) {
      dot += normalized_gradient[reduction_key] *
          normalized_k[q_base + reduction_key];
    }
    dot_shared = dot;
  }
  __syncthreads();
  const float key_projection = k_square_sum[row] > 1.0e-24f
      ? normalized_k[q_base + key] * dot_shared
      : 0.0f;
  dk[q_base + key] = __float2bfloat16_rn(
      k_inverse_norm[row] *
      (normalized_gradient[key] - key_projection));
  __syncthreads();

  if (key == 0) {
    float beta_gradient = 0.0f;
    for (int64_t tile = 0; tile < tile_count; ++tile) {
      const int64_t partial_offset =
          (((tile * batch + b) * chunk_capacity + local_token) * heads + h);
      beta_gradient += beta_tile_partial[partial_offset];
    }
    const float beta = chunk_sigmoid(
        __bfloat162float(beta_logits[row]));
    dbeta_logits[row] = __float2bfloat16_rn(
        beta_gradient * beta * (1.0f - beta));
  }

  float decay_gradient = 0.0f;
  for (int64_t tile = 0; tile < tile_count; ++tile) {
    const int64_t partial_offset =
        (((((tile * batch + b) * chunk_capacity + local_token) * heads + h) *
           key_dim) + key);
    decay_gradient += decay_tile_partial[partial_offset];
  }
  const float activated_gate = gate_sigmoid[q_base + key];
  const float activated_gradient = decay_gradient * lower_bound *
      activated_gate * (1.0f - activated_gate);
  const float raw_gradient = activated_gradient * expf(A_log[h]);
  draw_gate[q_base + key] = __float2bfloat16_rn(raw_gradient);
  // q/k partials are dead after this row finalization.  The existing
  // full-history normalized-adjoint allocation now records raw gate gradients
  // for deterministic parameter reductions after all reverse chunks finish.
  raw_gate_gradient_history[q_base + key] = raw_gradient;
}

__global__ void nanochat_kda_chunk_state_adjoint_init_128_kernel(
    const float* grad_final_state,
    float* state_adjoint,
    int64_t count) {
  const int64_t linear =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear < count) {
    state_adjoint[linear] = grad_final_state == nullptr
        ? 0.0f
        : grad_final_state[linear];
  }
}

__global__ void nanochat_kda_chunk_backward_parameter_finalize_128_kernel(
    const __nv_bfloat16* raw_gate,
    const float* dt_bias,
    const float* raw_gate_gradient_history,
    const float* state_adjoint,
    float* dA_log_partial,
    float* ddt_bias_partial,
    float* dinitial_state,
    int64_t batch,
    int64_t length,
    int64_t heads) {
  constexpr int64_t key_dim = 128;
  constexpr int64_t value_dim = 128;
  const int64_t recurrence = blockIdx.x;
  if (recurrence >= batch * heads) {
    return;
  }
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;
  const int64_t key = threadIdx.x;

  float bias_gradient = 0.0f;
  for (int64_t token = length; token-- > 0;) {
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    bias_gradient += raw_gate_gradient_history[q_base + key];
  }
  ddt_bias_partial[(b * heads + h) * key_dim + key] = bias_gradient;

  if (dinitial_state != nullptr) {
    for (int64_t value = 0; value < value_dim; ++value) {
      const int64_t state = state_index(
          b, h, value, key, heads, value_dim, key_dim);
      dinitial_state[state] = state_adjoint[state];
    }
  }

  if (key == 0) {
    float A_gradient = 0.0f;
    for (int64_t token = length; token-- > 0;) {
      const int64_t q_base = q_index(
          b, token, h, 0, length, heads, key_dim);
      for (int64_t reduction_key = 0;
           reduction_key < key_dim;
           ++reduction_key) {
        const float biased_gate =
            __bfloat162float(raw_gate[q_base + reduction_key]) +
            dt_bias[h * key_dim + reduction_key];
        A_gradient += raw_gate_gradient_history[q_base + reduction_key] *
            biased_gate;
      }
    }
    dA_log_partial[b * heads + h] = A_gradient;
  }
}

__global__ void nanochat_kda_chunk_backward_A_token_128_kernel(
    const __nv_bfloat16* raw_gate,
    const float* dt_bias,
    const float* raw_gate_gradient_history,
    float* A_token_partial,
    int64_t heads,
    int64_t row_count) {
  constexpr int64_t key_dim = 128;
  const int64_t row = blockIdx.x;
  if (row >= row_count) {
    return;
  }
  const int64_t h = row % heads;
  const int64_t key = threadIdx.x;
  const int64_t q_base = row * key_dim;
  __shared__ float contribution[128];

  contribution[key] = raw_gate_gradient_history[q_base + key] *
      (__bfloat162float(raw_gate[q_base + key]) +
       dt_bias[h * key_dim + key]);
  __syncthreads();
  if (key == 0) {
    float sum = 0.0f;
    for (int64_t reduction_key = 0;
         reduction_key < key_dim;
         ++reduction_key) {
      sum += contribution[reduction_key];
    }
    A_token_partial[row] = sum;
  }
}

__global__ void nanochat_kda_chunk_backward_parameter_reduce_128_kernel(
    const float* raw_gate_gradient_history,
    const float* A_token_partial,
    float* dA_log_partial,
    float* ddt_bias_partial,
    int64_t batch,
    int64_t length,
    int64_t heads) {
  constexpr int64_t key_dim = 128;
  const int64_t owner = blockIdx.x;
  if (owner >= batch * heads * key_dim || threadIdx.x != 0) {
    return;
  }
  const int64_t key = owner % key_dim;
  const int64_t recurrence = owner / key_dim;
  const int64_t h = recurrence % heads;
  const int64_t b = recurrence / heads;

  float bias_gradient = 0.0f;
  for (int64_t token = length; token-- > 0;) {
    const int64_t q_base = q_index(
        b, token, h, 0, length, heads, key_dim);
    bias_gradient += raw_gate_gradient_history[q_base + key];
  }
  ddt_bias_partial[(b * heads + h) * key_dim + key] = bias_gradient;

  if (key == 0) {
    float A_gradient = 0.0f;
    for (int64_t token = length; token-- > 0;) {
      const int64_t row = (b * length + token) * heads + h;
      A_gradient += A_token_partial[row];
    }
    dA_log_partial[b * heads + h] = A_gradient;
  }
}

__global__ void nanochat_kda_reduce_A_log_gradient_kernel(
    const float* partial,
    float* gradient,
    int64_t batch,
    int64_t heads) {
  const int64_t h = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (h >= heads) {
    return;
  }
  float sum = 0.0f;
  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    sum += partial[batch_index * heads + h];
  }
  gradient[h] = sum;
}

__global__ void nanochat_kda_reduce_dt_bias_gradient_kernel(
    const float* partial,
    float* gradient,
    int64_t batch,
    int64_t heads,
    int64_t key_dim) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count = heads * key_dim;
  if (linear >= count) {
    return;
  }
  const int64_t h = linear / key_dim;
  const int64_t key = linear - h * key_dim;
  float sum = 0.0f;
  for (int64_t batch_index = 0; batch_index < batch; ++batch_index) {
    sum += partial[(batch_index * heads + h) * key_dim + key];
  }
  gradient[linear] = sum;
}

void check_activation(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as q");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

void validate_chunk_inputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const c10::optional<at::Tensor>& initial_state) {
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16,
              "q must be CUDA bfloat16");
  TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, H, K]");
  check_activation(k, q, "k");
  check_activation(v, q, "v");
  check_activation(raw_gate, q, "raw_gate");
  check_activation(beta_logits, q, "beta_logits");
  TORCH_CHECK(k.sizes() == q.sizes() && raw_gate.sizes() == q.sizes(),
              "q, k, and raw_gate shapes must match");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == q.size(1) &&
                  v.size(2) == q.size(2),
              "v must match q in B, T, and H");
  TORCH_CHECK(beta_logits.dim() == 3 && beta_logits.size(0) == q.size(0) &&
                  beta_logits.size(1) == q.size(1) && beta_logits.size(2) == q.size(2),
              "beta_logits must have shape [B, T, H]");
  TORCH_CHECK(A_log.is_cuda() && A_log.device() == q.device() && A_log.is_contiguous() &&
                  A_log.scalar_type() == at::kFloat && A_log.numel() == q.size(2),
              "A_log must be contiguous CUDA float32 [H]");
  TORCH_CHECK(dt_bias.is_cuda() && dt_bias.device() == q.device() && dt_bias.is_contiguous() &&
                  dt_bias.scalar_type() == at::kFloat &&
                  dt_bias.numel() == q.size(2) * q.size(3),
              "dt_bias must be contiguous CUDA float32 with H*K elements");
  if (initial_state.has_value()) {
    const at::Tensor& state = *initial_state;
    TORCH_CHECK(state.is_cuda() && state.device() == q.device() && state.is_contiguous() &&
                    state.scalar_type() == at::kFloat,
                "initial_state must be contiguous CUDA float32");
    TORCH_CHECK(state.dim() == 4 && state.size(0) == q.size(0) &&
                    state.size(1) == q.size(2) && state.size(2) == v.size(3) &&
                    state.size(3) == q.size(3),
                "initial_state must have shape [B, H, V, K]");
  }
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> chunk_forward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const c10::optional<at::Tensor>& initial_state,
    bool output_final_state,
    double lower_bound,
    double scale) {
  validate_chunk_inputs(q, k, v, raw_gate, beta_logits, A_log, dt_bias, initial_state);
  const int64_t batch = q.size(0);
  const int64_t length = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);

  c10::cuda::CUDAGuard device_guard(q.device());
  const at::Tensor contiguous_q = q.contiguous();
  const at::Tensor contiguous_k = k.contiguous();
  const at::Tensor contiguous_v = v.contiguous();
  const at::Tensor contiguous_raw_gate = raw_gate.contiguous();
  const at::Tensor contiguous_beta_logits = beta_logits.contiguous();
  at::Tensor output = at::empty({batch, length, heads, value_dim}, v.options());
  at::Tensor state = at::empty(
      {batch, heads, value_dim, key_dim}, q.options().dtype(at::kFloat));
  const float* initial_pointer = initial_state.has_value()
      ? initial_state->data_ptr<float>()
      : nullptr;
  const int64_t count = batch * heads * value_dim;
  if (count > 0) {
    const int threads = 128;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
    if (key_dim == 128 && value_dim == 128) {
      nanochat_kda_chunk_forward_kernel<<<
          static_cast<int>(count), threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(contiguous_q.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_k.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_v.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_raw_gate.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_beta_logits.data_ptr<at::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          initial_pointer,
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          state.data_ptr<float>(),
          batch, length, heads,
          static_cast<float>(lower_bound), static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      const int blocks = static_cast<int>((count + threads - 1) / threads);
      nanochat_kda_chunk_forward_generic_kernel<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(contiguous_q.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_k.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_v.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_raw_gate.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_beta_logits.data_ptr<at::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          initial_pointer,
          reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
          state.data_ptr<float>(),
          batch, length, heads, key_dim, value_dim,
          static_cast<float>(lower_bound), static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  c10::optional<at::Tensor> final_state = c10::nullopt;
  if (output_final_state) {
    final_state = state;
  }
  return {output, final_state};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, c10::optional<at::Tensor>>
chunk_backward_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const c10::optional<at::Tensor>& initial_state,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& final_state,
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_final_state,
    double lower_bound,
    double scale) {
  validate_chunk_inputs(q, k, v, raw_gate, beta_logits, A_log, dt_bias, initial_state);
  check_activation(output, q, "output");
  check_activation(grad_output, q, "grad_output");
  TORCH_CHECK(output.sizes() == v.sizes() && grad_output.sizes() == v.sizes(),
              "output and grad_output must have shape [B, T, H, V]");
  const int64_t batch = q.size(0);
  const int64_t length = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  if (final_state.has_value()) {
    TORCH_CHECK(final_state->is_cuda() && final_state->device() == q.device() &&
                    final_state->is_contiguous() && final_state->scalar_type() == at::kFloat &&
                    final_state->sizes() == at::IntArrayRef({batch, heads, value_dim, key_dim}),
                "final_state must be contiguous CUDA float32 [B, H, V, K]");
  }
  const float* grad_final_pointer = nullptr;
  at::Tensor contiguous_grad_final;
  if (grad_final_state.has_value()) {
    TORCH_CHECK(final_state.has_value(), "grad_final_state requires final_state");
    TORCH_CHECK(grad_final_state->is_cuda() && grad_final_state->device() == q.device() &&
                    grad_final_state->scalar_type() == at::kFloat &&
                    grad_final_state->sizes() == at::IntArrayRef({batch, heads, value_dim, key_dim}),
                "grad_final_state must be CUDA float32 [B, H, V, K]");
    contiguous_grad_final = grad_final_state->contiguous();
    grad_final_pointer = contiguous_grad_final.data_ptr<float>();
  }
  const at::Tensor contiguous_grad_output = grad_output.contiguous();

  c10::cuda::CUDAGuard device_guard(q.device());
  const at::Tensor contiguous_q = q.contiguous();
  const at::Tensor contiguous_k = k.contiguous();
  const at::Tensor contiguous_v = v.contiguous();
  const at::Tensor contiguous_raw_gate = raw_gate.contiguous();
  const at::Tensor contiguous_beta_logits = beta_logits.contiguous();
  at::Tensor dq = at::empty(q.sizes(), q.options());
  at::Tensor dk = at::empty(k.sizes(), k.options());
  at::Tensor dv = at::empty(v.sizes(), v.options());
  at::Tensor draw_gate = at::empty(raw_gate.sizes(), raw_gate.options());
  at::Tensor dbeta_logits = at::empty(beta_logits.sizes(), beta_logits.options());
  at::Tensor dA_log = at::empty_like(A_log);
  at::Tensor ddt_bias = at::empty_like(dt_bias);
  at::Tensor dA_log_partial = at::empty({batch, heads}, A_log.options());
  at::Tensor ddt_bias_partial = at::empty(
      {batch, heads, key_dim}, A_log.options());
  c10::optional<at::Tensor> dinitial_state = c10::nullopt;
  float* dinitial_pointer = nullptr;
  if (initial_state.has_value()) {
    dinitial_state = at::empty_like(*initial_state);
    dinitial_pointer = dinitial_state->data_ptr<float>();
  }

  at::Tensor q_square_sum = at::empty({batch, length, heads}, A_log.options());
  at::Tensor k_square_sum = at::empty({batch, length, heads}, A_log.options());
  at::Tensor q_inverse_norm = at::empty({batch, length, heads}, A_log.options());
  at::Tensor k_inverse_norm = at::empty({batch, length, heads}, A_log.options());
  at::Tensor normalized_q = at::empty(q.sizes(), A_log.options());
  at::Tensor normalized_k = at::empty(k.sizes(), A_log.options());
  at::Tensor gate_sigmoid = at::empty(raw_gate.sizes(), A_log.options());
  at::Tensor decay_exponential = at::empty(raw_gate.sizes(), A_log.options());
  at::Tensor state_history = at::empty(
      {batch, heads, length + 1, value_dim, key_dim}, A_log.options());
  at::Tensor residual_history = at::empty(
      {batch, length, heads, value_dim}, A_log.options());
  at::Tensor state_adjoint = at::empty(
      {batch, heads, value_dim, key_dim}, A_log.options());
  at::Tensor normalized_adjoint = at::empty(q.sizes(), A_log.options());
  at::Tensor residual_adjoint = at::empty(
      {batch, heads, value_dim}, A_log.options());
  // Fixed-capacity production reverse workspaces.  Their sizes are independent
  // of T: three [G,B,Cmax,H,K] FP32 arrays plus one [G,B,Cmax,H].
  at::Tensor query_tile_partial;
  at::Tensor key_tile_partial;
  at::Tensor decay_tile_partial;
  at::Tensor beta_tile_partial;
  if (key_dim == 128 && value_dim == 128) {
    constexpr int64_t tile_count = 8;
    constexpr int64_t chunk_capacity = 32;
    query_tile_partial = at::empty(
        {tile_count, batch, chunk_capacity, heads, key_dim}, A_log.options());
    key_tile_partial = at::empty(
        {tile_count, batch, chunk_capacity, heads, key_dim}, A_log.options());
    decay_tile_partial = at::empty(
        {tile_count, batch, chunk_capacity, heads, key_dim}, A_log.options());
    beta_tile_partial = at::empty(
        {tile_count, batch, chunk_capacity, heads}, A_log.options());
  }

  const int threads = 1024;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
  const int64_t row_count = batch * length * heads;
  if (row_count > 0) {
    nanochat_kda_chunk_preprocess_kernel<<<
        static_cast<int>(row_count), threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(contiguous_q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(contiguous_k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(contiguous_raw_gate.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        q_square_sum.data_ptr<float>(),
        k_square_sum.data_ptr<float>(),
        q_inverse_norm.data_ptr<float>(),
        k_inverse_norm.data_ptr<float>(),
        normalized_q.data_ptr<float>(),
        normalized_k.data_ptr<float>(),
        gate_sigmoid.data_ptr<float>(),
        decay_exponential.data_ptr<float>(),
        heads, key_dim, row_count,
        static_cast<float>(lower_bound));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t recurrence_count = batch * heads;
  if (recurrence_count > 0) {
    const size_t shared_bytes = static_cast<size_t>(
        key_dim > value_dim ? key_dim : value_dim) * sizeof(float);
    if (key_dim == 128 && value_dim == 128) {
      constexpr int history_threads = 128;
      const int64_t history_count = recurrence_count * value_dim;
      nanochat_kda_chunk_history_128_kernel<<<
          static_cast<int>(history_count), history_threads, 0, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(contiguous_v.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_beta_logits.data_ptr<at::BFloat16>()),
          normalized_k.data_ptr<float>(),
          decay_exponential.data_ptr<float>(),
          initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr,
          state_history.data_ptr<float>(),
          residual_history.data_ptr<float>(),
          batch, length, heads, value_dim);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      const int64_t state_count = recurrence_count * value_dim * key_dim;
      const int state_blocks = static_cast<int>(
          (state_count + threads - 1) / threads);
      nanochat_kda_chunk_state_adjoint_init_128_kernel<<<
          state_blocks, threads, 0, stream>>>(
          grad_final_pointer,
          state_adjoint.data_ptr<float>(),
          state_count);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      constexpr int64_t tile_count = 8;
      constexpr int64_t chunk_capacity = 32;
      const int64_t chunk_count =
          (length + chunk_capacity - 1) / chunk_capacity;
      for (int64_t chunk = chunk_count; chunk-- > 0;) {
        const int64_t chunk_start = chunk * chunk_capacity;
        const int64_t chunk_end =
            (chunk_start + chunk_capacity < length)
            ? chunk_start + chunk_capacity
            : length;
        const int64_t chunk_length = chunk_end - chunk_start;
        nanochat_kda_chunk_backward_kernel<<<
            static_cast<int>(recurrence_count * tile_count),
            threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_beta_logits.data_ptr<at::BFloat16>()),
            normalized_q.data_ptr<float>(),
            normalized_k.data_ptr<float>(),
            decay_exponential.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_grad_output.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
            state_history.data_ptr<float>(),
            residual_history.data_ptr<float>(),
            state_adjoint.data_ptr<float>(),
            residual_adjoint.data_ptr<float>(),
            query_tile_partial.data_ptr<float>(),
            key_tile_partial.data_ptr<float>(),
            decay_tile_partial.data_ptr<float>(),
            beta_tile_partial.data_ptr<float>(),
            batch, length, heads, chunk_start, chunk_end,
            static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        constexpr int finalize_threads = 128;
        nanochat_kda_chunk_backward_row_finalize_128_kernel<<<
            static_cast<int>(recurrence_count * chunk_length),
            finalize_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_beta_logits.data_ptr<at::BFloat16>()),
            A_log.data_ptr<float>(),
            q_square_sum.data_ptr<float>(),
            k_square_sum.data_ptr<float>(),
            q_inverse_norm.data_ptr<float>(),
            k_inverse_norm.data_ptr<float>(),
            normalized_q.data_ptr<float>(),
            normalized_k.data_ptr<float>(),
            gate_sigmoid.data_ptr<float>(),
            query_tile_partial.data_ptr<float>(),
            key_tile_partial.data_ptr<float>(),
            decay_tile_partial.data_ptr<float>(),
            beta_tile_partial.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dq.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(draw_gate.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(
                dbeta_logits.data_ptr<at::BFloat16>()),
            normalized_adjoint.data_ptr<float>(),
            batch, length, heads, chunk_start, chunk_length,
            static_cast<float>(lower_bound), static_cast<float>(scale));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }

      if (row_count > 0) {
        constexpr int A_token_threads = 128;
        nanochat_kda_chunk_backward_A_token_128_kernel<<<
            static_cast<int>(row_count),
            A_token_threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                contiguous_raw_gate.data_ptr<at::BFloat16>()),
            dt_bias.data_ptr<float>(),
            normalized_adjoint.data_ptr<float>(),
            q_square_sum.data_ptr<float>(),
            heads, row_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }

      constexpr int parameter_threads = 1;
      const int64_t parameter_owner_count = recurrence_count * key_dim;
      nanochat_kda_chunk_backward_parameter_reduce_128_kernel<<<
          static_cast<int>(parameter_owner_count),
          parameter_threads, 0, stream>>>(
          normalized_adjoint.data_ptr<float>(),
          q_square_sum.data_ptr<float>(),
          dA_log_partial.data_ptr<float>(),
          ddt_bias_partial.data_ptr<float>(),
          batch, length, heads);
      C10_CUDA_KERNEL_LAUNCH_CHECK();

      if (dinitial_pointer != nullptr) {
        nanochat_kda_chunk_state_adjoint_init_128_kernel<<<
            state_blocks, threads, 0, stream>>>(
            state_adjoint.data_ptr<float>(),
            dinitial_pointer,
            state_count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      nanochat_kda_chunk_backward_generic_kernel<<<
          static_cast<int>(recurrence_count), threads, shared_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(contiguous_v.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_raw_gate.data_ptr<at::BFloat16>()),
          reinterpret_cast<const __nv_bfloat16*>(contiguous_beta_logits.data_ptr<at::BFloat16>()),
          A_log.data_ptr<float>(),
          dt_bias.data_ptr<float>(),
          q_square_sum.data_ptr<float>(),
          k_square_sum.data_ptr<float>(),
          q_inverse_norm.data_ptr<float>(),
          k_inverse_norm.data_ptr<float>(),
          normalized_q.data_ptr<float>(),
          normalized_k.data_ptr<float>(),
          gate_sigmoid.data_ptr<float>(),
          decay_exponential.data_ptr<float>(),
          initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr,
          reinterpret_cast<const __nv_bfloat16*>(
              contiguous_grad_output.data_ptr<at::BFloat16>()),
          grad_final_pointer,
          reinterpret_cast<__nv_bfloat16*>(dq.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(draw_gate.data_ptr<at::BFloat16>()),
          reinterpret_cast<__nv_bfloat16*>(dbeta_logits.data_ptr<at::BFloat16>()),
          dA_log_partial.data_ptr<float>(),
          ddt_bias_partial.data_ptr<float>(),
          dinitial_pointer,
          state_history.data_ptr<float>(),
          residual_history.data_ptr<float>(),
          state_adjoint.data_ptr<float>(),
          normalized_adjoint.data_ptr<float>(),
          residual_adjoint.data_ptr<float>(),
          batch, length, heads, key_dim, value_dim,
          static_cast<float>(lower_bound), static_cast<float>(scale));
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  if (heads > 0) {
    const int blocks = static_cast<int>((heads + threads - 1) / threads);
    nanochat_kda_reduce_A_log_gradient_kernel<<<blocks, threads, 0, stream>>>(
        dA_log_partial.data_ptr<float>(), dA_log.data_ptr<float>(), batch, heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int64_t dt_bias_count = heads * key_dim;
  if (dt_bias_count > 0) {
    const int blocks = static_cast<int>((dt_bias_count + threads - 1) / threads);
    nanochat_kda_reduce_dt_bias_gradient_kernel<<<blocks, threads, 0, stream>>>(
        ddt_bias_partial.data_ptr<float>(), ddt_bias.data_ptr<float>(),
        batch, heads, key_dim);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dq, dk, dv, draw_gate, dbeta_logits, dA_log, ddt_bias, dinitial_state};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(nanochat_kda, m) {
  m.def("chunk_forward(Tensor q, Tensor k, Tensor v, Tensor raw_gate, "
        "Tensor beta_logits, Tensor A_log, Tensor dt_bias, Tensor? initial_state, "
        "bool output_final_state, float lower_bound, float scale) "
        "-> (Tensor output, Tensor? final_state)");
  m.def("chunk_backward(Tensor q, Tensor k, Tensor v, Tensor raw_gate, "
        "Tensor beta_logits, Tensor A_log, Tensor dt_bias, Tensor? initial_state, "
        "Tensor output, Tensor? final_state, Tensor grad_output, "
        "Tensor? grad_final_state, float lower_bound, float scale) -> "
        "(Tensor dq, Tensor dk, Tensor dv, Tensor draw_gate, "
        "Tensor dbeta_logits, Tensor dA_log, Tensor ddt_bias, "
        "Tensor? dinitial_state)");
}

TORCH_LIBRARY_IMPL(nanochat_kda, CUDA, m) {
  m.impl("chunk_forward", &chunk_forward_cuda);
  m.impl("chunk_backward", &chunk_backward_cuda);
}
