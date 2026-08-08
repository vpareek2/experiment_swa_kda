#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <tuple>

namespace {

__device__ float sigmoidf_stable(float x) {
  if (x >= 0.0f) {
    const float z = expf(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = expf(x);
  return z / (1.0f + z);
}

__global__ void nanochat_kda_recurrent_decode_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* raw_gate,
    const __nv_bfloat16* beta_logits,
    const float* A_log,
    const float* dt_bias,
    const float* initial_state,
    __nv_bfloat16* output,
    float* final_state,
    int64_t batch,
    int64_t heads,
    int64_t key_dim,
    int64_t value_dim,
    float lower_bound,
    float scale) {
  const int64_t bh = blockIdx.x;
  if (bh >= batch * heads) {
    return;
  }
  const int64_t b = bh / heads;
  const int64_t h = bh - b * heads;
  const int64_t q_base = (b * heads + h) * key_dim;
  const int64_t v_base = (b * heads + h) * value_dim;

  float q_square_sum = 0.0f;
  float k_square_sum = 0.0f;
  for (int64_t key = 0; key < key_dim; ++key) {
    const float q_value = __bfloat162float(q[q_base + key]);
    const float k_value = __bfloat162float(k[q_base + key]);
    q_square_sum += q_value * q_value;
    k_square_sum += k_value * k_value;
  }
  const float q_inverse_norm = rsqrtf(fmaxf(q_square_sum, 1.0e-24f));
  const float k_inverse_norm = rsqrtf(fmaxf(k_square_sum, 1.0e-24f));
  const float beta = sigmoidf_stable(__bfloat162float(beta_logits[bh]));
  const float a = expf(A_log[h]);

  for (int64_t value_index = threadIdx.x; value_index < value_dim;
       value_index += blockDim.x) {
    const int64_t state_base = ((b * heads + h) * value_dim + value_index) * key_dim;
    float prediction = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float normalized_key = __bfloat162float(k[q_base + key]) * k_inverse_norm;
      const float gate_input = __bfloat162float(raw_gate[q_base + key]);
      const float decay = lower_bound *
          sigmoidf_stable(a * (gate_input + dt_bias[h * key_dim + key]));
      const float old_state = initial_state == nullptr ? 0.0f : initial_state[state_base + key];
      prediction += normalized_key * (old_state * expf(decay));
    }

    const float residual = __bfloat162float(v[v_base + value_index]) - prediction;
    float output_value = 0.0f;
    for (int64_t key = 0; key < key_dim; ++key) {
      const float normalized_key = __bfloat162float(k[q_base + key]) * k_inverse_norm;
      const float normalized_query =
          __bfloat162float(q[q_base + key]) * q_inverse_norm * scale;
      const float gate_input = __bfloat162float(raw_gate[q_base + key]);
      const float decay = lower_bound *
          sigmoidf_stable(a * (gate_input + dt_bias[h * key_dim + key]));
      const float old_state = initial_state == nullptr ? 0.0f : initial_state[state_base + key];
      const float updated_state = old_state * expf(decay) + beta * normalized_key * residual;
      output_value += normalized_query * updated_state;
      if (final_state != nullptr) {
        final_state[state_base + key] = updated_state;
      }
    }
    output[v_base + value_index] = __float2bfloat16_rn(output_value);
  }
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> recurrent_decode_cuda(
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
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                  raw_gate.is_contiguous() && beta_logits.is_contiguous() &&
                  A_log.is_contiguous() && dt_bias.is_contiguous(),
              "all recurrent decode inputs must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16 &&
                  raw_gate.scalar_type() == at::kBFloat16 &&
                  beta_logits.scalar_type() == at::kBFloat16,
              "activation tensors must be bfloat16");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat && dt_bias.scalar_type() == at::kFloat,
              "A_log and dt_bias must be float32");
  TORCH_CHECK(q.dim() == 4 && q.size(1) == 1, "recurrent decode requires T=1");
  TORCH_CHECK(k.sizes() == q.sizes() && raw_gate.sizes() == q.sizes(),
              "q, k, and raw_gate shapes must match");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) && v.size(1) == 1 &&
                  v.size(2) == q.size(2),
              "v shape must match B, T, and H");
  TORCH_CHECK(beta_logits.dim() == 3 && beta_logits.size(0) == q.size(0) &&
                  beta_logits.size(1) == 1 && beta_logits.size(2) == q.size(2),
              "beta_logits must have shape [B, 1, H]");

  const int64_t batch = q.size(0);
  const int64_t heads = q.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  TORCH_CHECK(A_log.numel() == heads && dt_bias.numel() == heads * key_dim,
              "gate parameter shapes do not match H and K");

  const float* initial_pointer = nullptr;
  if (initial_state.has_value()) {
    const at::Tensor& state = *initial_state;
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat,
                "initial_state must be contiguous CUDA float32");
    TORCH_CHECK(state.dim() == 4 && state.size(0) == batch && state.size(1) == heads &&
                    state.size(2) == value_dim && state.size(3) == key_dim,
                "initial_state must have shape [B, H, V, K]");
    initial_pointer = state.data_ptr<float>();
  }

  c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor output = at::empty({batch, 1, heads, value_dim}, v.options());
  c10::optional<at::Tensor> final_state = c10::nullopt;
  float* final_pointer = nullptr;
  if (output_final_state) {
    final_state = at::empty({batch, heads, value_dim, key_dim},
                            q.options().dtype(at::kFloat));
    final_pointer = final_state->data_ptr<float>();
  }

  const int threads = 128;
  const int blocks = static_cast<int>(batch * heads);
  nanochat_kda_recurrent_decode_kernel<<<blocks, threads, 0,
      at::cuda::getCurrentCUDAStream(q.get_device())>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(raw_gate.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(beta_logits.data_ptr<at::BFloat16>()),
      A_log.data_ptr<float>(),
      dt_bias.data_ptr<float>(),
      initial_pointer,
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      final_pointer,
      batch,
      heads,
      key_dim,
      value_dim,
      static_cast<float>(lower_bound),
      static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, final_state};
}

}  // namespace

TORCH_LIBRARY(nanochat_kda, m) {
  m.def("recurrent_decode(Tensor q, Tensor k, Tensor v, Tensor raw_gate, "
        "Tensor beta_logits, Tensor A_log, Tensor dt_bias, Tensor? initial_state, "
        "bool output_final_state, float lower_bound, float scale) "
        "-> (Tensor output, Tensor? final_state)");
}

TORCH_LIBRARY_IMPL(nanochat_kda, CUDA, m) {
  m.impl("recurrent_decode", &recurrent_decode_cuda);
}
