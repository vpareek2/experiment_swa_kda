#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

__device__ __forceinline__ float stable_sigmoid(float value) {
  if (value >= 0.0f) {
    const float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float convolution_source_value_backward(
    const __nv_bfloat16* x,
    const __nv_bfloat16* initial_state,
    int64_t b,
    int64_t c,
    int64_t source_index,
    int64_t length,
    int64_t channels,
    int64_t width) {
  if (source_index >= 0) {
    return __bfloat162float(x[(b * length + source_index) * channels + c]);
  }
  if (initial_state == nullptr) {
    return 0.0f;
  }
  return __bfloat162float(
      initial_state[(b * channels + c) * width + width + source_index]);
}

__device__ __forceinline__ float preactivation_gradient(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* initial_state,
    const __nv_bfloat16* grad_output,
    int64_t b,
    int64_t token_index,
    int64_t c,
    int64_t length,
    int64_t channels,
    int64_t width) {
  float preactivation = 0.0f;
  for (int64_t tap = 0; tap < width; ++tap) {
    const int64_t source_index = token_index + tap - (width - 1);
    const float source = convolution_source_value_backward(
        x, initial_state, b, c, source_index, length, channels, width);
    const float product = source * __bfloat162float(weight[c * width + tap]);
    preactivation += __bfloat162float(__float2bfloat16_rn(product));
  }
  const float rounded_preactivation =
      __bfloat162float(__float2bfloat16_rn(preactivation));
  const float sigmoid = stable_sigmoid(rounded_preactivation);
  const float silu_derivative =
      sigmoid * (1.0f + rounded_preactivation * (1.0f - sigmoid));
  const int64_t output_index = (b * length + token_index) * channels + c;
  return __bfloat162float(grad_output[output_index]) * silu_derivative;
}

__global__ void nanochat_kda_causal_convolution_backward_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* initial_state,
    const __nv_bfloat16* grad_output,
    const __nv_bfloat16* grad_final_state,
    __nv_bfloat16* dx,
    __nv_bfloat16* dweight,
    __nv_bfloat16* dinitial_state,
    int64_t batch,
    int64_t length,
    int64_t channels,
    int64_t width,
    int64_t dx_elements,
    int64_t weight_elements,
    int64_t state_elements,
    int64_t total_elements) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total_elements) {
    return;
  }

  if (index < dx_elements) {
    const int64_t c = index % channels;
    const int64_t input_index = (index / channels) % length;
    const int64_t b = index / (length * channels);
    float gradient = 0.0f;
    const int64_t remaining_tokens = length - input_index;
    const int64_t last_token =
        width < remaining_tokens ? input_index + width : length;
    for (int64_t token_index = input_index; token_index < last_token;
         ++token_index) {
      const int64_t tap = input_index - token_index + width - 1;
      gradient += preactivation_gradient(
          x, weight, initial_state, grad_output, b, token_index, c,
          length, channels, width) *
          __bfloat162float(weight[c * width + tap]);
    }
    if (grad_final_state != nullptr) {
      const int64_t final_index = input_index - length + width;
      if (final_index >= 0 && final_index < width) {
        gradient += __bfloat162float(
            grad_final_state[(b * channels + c) * width + final_index]);
      }
    }
    dx[index] = __float2bfloat16_rn(gradient);
    return;
  }

  const int64_t shifted = index - dx_elements;
  if (shifted < weight_elements) {
    const int64_t tap = shifted % width;
    const int64_t c = shifted / width;
    float gradient = 0.0f;
    for (int64_t b = 0; b < batch; ++b) {
      for (int64_t token_index = 0; token_index < length; ++token_index) {
        const int64_t source_index = token_index + tap - (width - 1);
        const float source = convolution_source_value_backward(
            x, initial_state, b, c, source_index, length, channels, width);
        gradient += preactivation_gradient(
            x, weight, initial_state, grad_output, b, token_index, c,
            length, channels, width) * source;
      }
    }
    dweight[shifted] = __float2bfloat16_rn(gradient);
    return;
  }

  const int64_t state_offset = shifted - weight_elements;
  if (state_offset < state_elements) {
    const int64_t state_index = state_offset % width;
    const int64_t c = (state_offset / width) % channels;
    const int64_t b = state_offset / (channels * width);
    float gradient = 0.0f;
    const int64_t last_token = state_index < length ? state_index : length;
    for (int64_t token_index = 0; token_index < last_token; ++token_index) {
      const int64_t tap = state_index - token_index - 1;
      gradient += preactivation_gradient(
          x, weight, initial_state, grad_output, b, token_index, c,
          length, channels, width) *
          __bfloat162float(weight[c * width + tap]);
    }
    if (grad_final_state != nullptr) {
      const int64_t final_index = state_index - length;
      if (final_index >= 0 && final_index < width) {
        gradient += __bfloat162float(
            grad_final_state[(b * channels + c) * width + final_index]);
      }
    }
    dinitial_state[state_offset] = __float2bfloat16_rn(gradient);
  }
}

void check_backward_tensor(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as x");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

void check_state_shape(
    const at::Tensor& tensor,
    int64_t batch,
    int64_t channels,
    int64_t width,
    const char* name) {
  TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == batch &&
                  tensor.size(1) == channels && tensor.size(2) == width,
              name, " must have shape [B, C, W]");
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
causal_convolution_backward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& initial_state,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& final_state,
    const at::Tensor& grad_output,
    const c10::optional<at::Tensor>& grad_final_state) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                  x.scalar_type() == at::kBFloat16 && x.dim() == 3,
              "x must be contiguous CUDA bfloat16 [B, T, C]");
  check_backward_tensor(weight, x, "weight");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == x.size(2) && weight.size(1) > 0,
              "weight must have shape [C, W]");
  const int64_t batch = x.size(0);
  const int64_t length = x.size(1);
  const int64_t channels = x.size(2);
  const int64_t width = weight.size(1);

  check_backward_tensor(output, x, "output");
  check_backward_tensor(grad_output, x, "grad_output");
  TORCH_CHECK(output.sizes() == x.sizes(), "output must have shape [B, T, C]");
  TORCH_CHECK(grad_output.sizes() == x.sizes(),
              "grad_output must have shape [B, T, C]");

  const __nv_bfloat16* initial_pointer = nullptr;
  if (initial_state.has_value()) {
    check_backward_tensor(*initial_state, x, "initial_state");
    check_state_shape(*initial_state, batch, channels, width, "initial_state");
    initial_pointer = reinterpret_cast<const __nv_bfloat16*>(
        initial_state->data_ptr<at::BFloat16>());
  }
  if (final_state.has_value()) {
    check_backward_tensor(*final_state, x, "final_state");
    check_state_shape(*final_state, batch, channels, width, "final_state");
  }
  const __nv_bfloat16* grad_final_pointer = nullptr;
  if (grad_final_state.has_value()) {
    TORCH_CHECK(final_state.has_value(),
                "grad_final_state requires a forward final_state");
    check_backward_tensor(*grad_final_state, x, "grad_final_state");
    check_state_shape(*grad_final_state, batch, channels, width, "grad_final_state");
    grad_final_pointer = reinterpret_cast<const __nv_bfloat16*>(
        grad_final_state->data_ptr<at::BFloat16>());
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor dx = at::empty_like(x);
  at::Tensor dweight = at::empty_like(weight);
  c10::optional<at::Tensor> dinitial_state = c10::nullopt;
  __nv_bfloat16* dinitial_pointer = nullptr;
  const int64_t dx_elements = batch * length * channels;
  const int64_t weight_elements = channels * width;
  int64_t state_elements = 0;
  if (initial_state.has_value()) {
    dinitial_state = at::empty_like(*initial_state);
    dinitial_pointer = reinterpret_cast<__nv_bfloat16*>(
        dinitial_state->data_ptr<at::BFloat16>());
    state_elements = batch * channels * width;
  }
  const int64_t total_elements = dx_elements + weight_elements + state_elements;
  if (total_elements > 0) {
    const int threads = 256;
    const int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    nanochat_kda_causal_convolution_backward_kernel<<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream(x.get_device())>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        initial_pointer,
        reinterpret_cast<const __nv_bfloat16*>(grad_output.data_ptr<at::BFloat16>()),
        grad_final_pointer,
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dweight.data_ptr<at::BFloat16>()),
        dinitial_pointer,
        batch,
        length,
        channels,
        width,
        dx_elements,
        weight_elements,
        state_elements,
        total_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dweight, dinitial_state};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(nanochat_kda, m) {
  m.def("causal_convolution_backward(Tensor x, Tensor weight, Tensor? initial_state, "
        "Tensor output, Tensor? final_state, Tensor grad_output, "
        "Tensor? grad_final_state) -> "
        "(Tensor dx, Tensor dweight, Tensor? dinitial_state)");
}

TORCH_LIBRARY_IMPL(nanochat_kda, CUDA, m) {
  m.impl("causal_convolution_backward", &causal_convolution_backward_cuda);
}
