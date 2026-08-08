#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <tuple>

namespace {

__device__ __forceinline__ float stable_sigmoid_forward(float value) {
  if (value >= 0.0f) {
    const float exponential = expf(-value);
    return 1.0f / (1.0f + exponential);
  }
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

__device__ __forceinline__ float convolution_source_value(
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
  const int64_t state_index = width + source_index;
  return __bfloat162float(initial_state[(b * channels + c) * width + state_index]);
}

__global__ void nanochat_kda_causal_convolution_forward_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* initial_state,
    __nv_bfloat16* output,
    int64_t length,
    int64_t channels,
    int64_t width,
    int64_t element_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const int64_t c = index % channels;
  const int64_t token_index = (index / channels) % length;
  const int64_t b = index / (length * channels);

  float preactivation = 0.0f;
  for (int64_t tap = 0; tap < width; ++tap) {
    const int64_t source_index = token_index + tap - (width - 1);
    const float source = convolution_source_value(
        x, initial_state, b, c, source_index, length, channels, width);
    const float product = source * __bfloat162float(weight[c * width + tap]);
    preactivation += __bfloat162float(__float2bfloat16_rn(product));
  }
  const float rounded_preactivation =
      __bfloat162float(__float2bfloat16_rn(preactivation));
  const float sigmoid = stable_sigmoid_forward(rounded_preactivation);
  output[index] = __float2bfloat16_rn(rounded_preactivation * sigmoid);
}

__global__ void nanochat_kda_causal_convolution_final_state_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* initial_state,
    __nv_bfloat16* final_state,
    int64_t length,
    int64_t channels,
    int64_t width,
    int64_t element_count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= element_count) {
    return;
  }
  const int64_t state_index = index % width;
  const int64_t c = (index / width) % channels;
  const int64_t b = index / (channels * width);
  const int64_t source_index = length + state_index - width;
  final_state[index] = __float2bfloat16_rn(convolution_source_value(
      x, initial_state, b, c, source_index, length, channels, width));
}

void check_convolution_tensor(
    const at::Tensor& tensor,
    const at::Tensor& reference,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as x");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be bfloat16");
}

std::tuple<at::Tensor, c10::optional<at::Tensor>> causal_convolution_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& initial_state,
    bool output_final_state) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B, T, C]");
  check_convolution_tensor(weight, x, "weight");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) == x.size(2),
              "weight must have shape [C, W]");
  TORCH_CHECK(weight.size(1) > 0, "convolution width must be positive");

  const int64_t batch = x.size(0);
  const int64_t length = x.size(1);
  const int64_t channels = x.size(2);
  const int64_t width = weight.size(1);
  const __nv_bfloat16* initial_pointer = nullptr;
  if (initial_state.has_value()) {
    const at::Tensor& state = *initial_state;
    check_convolution_tensor(state, x, "initial_state");
    TORCH_CHECK(state.dim() == 3 && state.size(0) == batch &&
                    state.size(1) == channels && state.size(2) == width,
                "initial_state must have shape [B, C, W]");
    initial_pointer = reinterpret_cast<const __nv_bfloat16*>(
        state.data_ptr<at::BFloat16>());
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor output = at::empty({batch, length, channels}, x.options());
  c10::optional<at::Tensor> final_state = c10::nullopt;
  const int threads = 256;
  const int64_t output_elements = batch * length * channels;
  if (output_elements > 0) {
    const int blocks = static_cast<int>((output_elements + threads - 1) / threads);
    nanochat_kda_causal_convolution_forward_kernel<<<blocks, threads, 0,
        at::cuda::getCurrentCUDAStream(x.get_device())>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        initial_pointer,
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        length,
        channels,
        width,
        output_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  if (output_final_state) {
    final_state = at::empty({batch, channels, width}, x.options());
    const int64_t state_elements = batch * channels * width;
    if (state_elements > 0) {
      const int blocks = static_cast<int>((state_elements + threads - 1) / threads);
      nanochat_kda_causal_convolution_final_state_kernel<<<blocks, threads, 0,
          at::cuda::getCurrentCUDAStream(x.get_device())>>>(
          reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
          initial_pointer,
          reinterpret_cast<__nv_bfloat16*>(final_state->data_ptr<at::BFloat16>()),
          length,
          channels,
          width,
          state_elements);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }
  return {output, final_state};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(nanochat_kda, m) {
  m.def("causal_convolution_forward(Tensor x, Tensor weight, Tensor? initial_state, "
        "bool output_final_state) -> (Tensor output, Tensor? final_state)");
}

TORCH_LIBRARY_IMPL(nanochat_kda, CUDA, m) {
  m.impl("causal_convolution_forward", &causal_convolution_forward_cuda);
}
