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

constexpr int kDim = 128;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ float stable_sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void nanochat_kda_rmsnorm_gate_forward_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate,
    const float* weight,
    __nv_bfloat16* output,
    float* inverse_rms,
    int64_t rows,
    float epsilon) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * kDim;
  float square_sum = 0.0f;
  float x_values[4];
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int d = lane * 4 + item;
    const float value = __bfloat162float(x[base + d]);
    x_values[item] = value;
    square_sum += value * value;
  }
  square_sum = warp_sum(square_sum);
  const float inverse = rsqrtf(square_sum / static_cast<float>(kDim) + epsilon);
  if (lane == 0) {
    inverse_rms[row] = inverse;
  }
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int d = lane * 4 + item;
    const float rounded_weight = __bfloat162float(__float2bfloat16_rn(weight[d]));
    const __nv_bfloat16 normalized =
        __float2bfloat16_rn(x_values[item] * inverse * rounded_weight);
    const __nv_bfloat16 sigmoid_gate = __float2bfloat16_rn(
        stable_sigmoid(__bfloat162float(gate[base + d])));
    output[base + d] = __float2bfloat16_rn(
        __bfloat162float(normalized) * __bfloat162float(sigmoid_gate));
  }
}

__global__ void nanochat_kda_rmsnorm_gate_backward_kernel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* gate,
    const float* weight,
    const float* inverse_rms,
    const __nv_bfloat16* grad_output,
    __nv_bfloat16* dx,
    __nv_bfloat16* dgate,
    float* dweight_partial,
    int64_t rows) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
  __shared__ float weight_contribution[kWarpsPerBlock][kDim];

  float x_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float grad_normalized[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float rounded_weight[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float inverse = 0.0f;
  float dot = 0.0f;
  if (row < rows) {
    const int64_t base = row * kDim;
    inverse = inverse_rms[row];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int d = lane * 4 + item;
      const float x_value = __bfloat162float(x[base + d]);
      const float gate_value = __bfloat162float(gate[base + d]);
      const float dy = __bfloat162float(grad_output[base + d]);
      const float w = __bfloat162float(__float2bfloat16_rn(weight[d]));
      const __nv_bfloat16 sigmoid_gate_bf =
          __float2bfloat16_rn(stable_sigmoid(gate_value));
      const float sigmoid_gate = __bfloat162float(sigmoid_gate_bf);
      const __nv_bfloat16 normalized_bf =
          __float2bfloat16_rn(x_value * inverse * w);
      const float normalized = __bfloat162float(normalized_bf);
      const float grad_norm = __bfloat162float(
          __float2bfloat16_rn(dy * sigmoid_gate));
      const float grad_sigmoid = __bfloat162float(
          __float2bfloat16_rn(dy * normalized));
      x_values[item] = x_value;
      grad_normalized[item] = grad_norm;
      rounded_weight[item] = w;
      dot += grad_norm * w * x_value;
      const __nv_bfloat16 one_minus_sigmoid =
          __float2bfloat16_rn(1.0f - sigmoid_gate);
      const __nv_bfloat16 sigmoid_backward_intermediate =
          __float2bfloat16_rn(
              grad_sigmoid * __bfloat162float(one_minus_sigmoid));
      dgate[base + d] = __float2bfloat16_rn(
          __bfloat162float(sigmoid_backward_intermediate) * sigmoid_gate);
      weight_contribution[warp][d] = grad_norm * x_value * inverse;
    }
  } else {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      weight_contribution[warp][lane * 4 + item] = 0.0f;
    }
  }
  dot = warp_sum(dot);
  if (row < rows) {
    const int64_t base = row * kDim;
    const float correction = dot * inverse * inverse / static_cast<float>(kDim);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const int d = lane * 4 + item;
      dx[base + d] = __float2bfloat16_rn(
          inverse * (grad_normalized[item] * rounded_weight[item] -
                     x_values[item] * correction));
    }
  }
  __syncthreads();
  if (threadIdx.x < kDim) {
    float partial = 0.0f;
#pragma unroll
    for (int row_warp = 0; row_warp < kWarpsPerBlock; ++row_warp) {
      partial += weight_contribution[row_warp][threadIdx.x];
    }
    dweight_partial[static_cast<int64_t>(blockIdx.x) * kDim + threadIdx.x] =
        partial;
  }
}

__global__ void nanochat_kda_rmsnorm_gate_reduce_weight_kernel(
    const float* partial,
    float* dweight,
    int blocks) {
  const int d = blockIdx.x;
  float sum = 0.0f;
  for (int index = threadIdx.x; index < blocks; index += blockDim.x) {
    sum += partial[static_cast<int64_t>(index) * kDim + d];
  }
  sum = warp_sum(sum);
  __shared__ float warp_partials[8];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (lane == 0) {
    warp_partials[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    float block_sum = lane < 8 ? warp_partials[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      // The public parameter is FP32, but the current module casts it to BF16
      // before RMSNorm; its autograd gradient crosses the inverse cast.
      dweight[d] = __bfloat162float(__float2bfloat16_rn(block_sum));
    }
  }
}

void validate(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight) {
  TORCH_CHECK(x.is_cuda() && gate.is_cuda() && weight.is_cuda() &&
              x.device() == gate.device() && x.device() == weight.device(),
              "rmsnorm gate tensors must share one CUDA device");
  TORCH_CHECK(x.is_contiguous() && gate.is_contiguous() && weight.is_contiguous(),
              "rmsnorm gate tensors must be contiguous");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
              gate.scalar_type() == at::kBFloat16,
              "rmsnorm gate activations must be BF16");
  TORCH_CHECK(weight.scalar_type() == at::kFloat,
              "rmsnorm gate weight must be FP32");
  TORCH_CHECK(x.sizes() == gate.sizes() && x.dim() >= 1 &&
              x.size(-1) == kDim && weight.numel() == kDim,
              "rmsnorm gate requires matching [...,128] activations and weight");
}

std::tuple<at::Tensor, at::Tensor> rmsnorm_gate_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double epsilon) {
  validate(x, gate, weight);
  TORCH_CHECK(epsilon > 0.0 && std::isfinite(epsilon),
              "rmsnorm gate epsilon must be positive and finite");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor output = at::empty_like(x);
  const int64_t rows = x.numel() / kDim;
  at::Tensor inverse = at::empty({rows}, weight.options());
  const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) /
                                      kWarpsPerBlock);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (rows > 0) {
    nanochat_kda_rmsnorm_gate_forward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
        weight.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        inverse.data_ptr<float>(), rows, static_cast<float>(epsilon));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output, inverse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> rmsnorm_gate_backward_cuda(
    const at::Tensor& x,
    const at::Tensor& gate,
    const at::Tensor& weight,
    const at::Tensor& inverse,
    const at::Tensor& grad_output,
    double epsilon) {
  (void)epsilon;
  validate(x, gate, weight);
  TORCH_CHECK(grad_output.is_cuda() && grad_output.device() == x.device() &&
              grad_output.is_contiguous() &&
              grad_output.scalar_type() == at::kBFloat16 &&
              grad_output.sizes() == x.sizes(),
              "rmsnorm gate grad_output must be contiguous CUDA BF16 matching x");
  const int64_t rows = x.numel() / kDim;
  TORCH_CHECK(inverse.is_cuda() && inverse.device() == x.device() &&
              inverse.is_contiguous() && inverse.scalar_type() == at::kFloat &&
              inverse.numel() == rows,
              "rmsnorm gate inverse must be contiguous CUDA FP32 [rows]");
  c10::cuda::CUDAGuard guard(x.device());
  at::Tensor dx = at::empty_like(x);
  at::Tensor dgate = at::empty_like(gate);
  at::Tensor dweight = at::empty_like(weight);
  const int blocks = static_cast<int>((rows + kWarpsPerBlock - 1) /
                                      kWarpsPerBlock);
  at::Tensor partial = at::empty({blocks, kDim}, weight.options());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (rows > 0) {
    nanochat_kda_rmsnorm_gate_backward_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr<at::BFloat16>()),
        weight.data_ptr<float>(), inverse.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(
            grad_output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dx.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dgate.data_ptr<at::BFloat16>()),
        partial.data_ptr<float>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    nanochat_kda_rmsnorm_gate_reduce_weight_kernel<<<kDim, 256, 0, stream>>>(
        partial.data_ptr<float>(), dweight.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {dx, dgate, dweight};
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(nanochat_kda, m) {
  m.def("rmsnorm_gate_forward(Tensor x, Tensor gate, Tensor weight, float epsilon) "
        "-> (Tensor output, Tensor inverse_rms)");
  m.def("rmsnorm_gate_backward(Tensor x, Tensor gate, Tensor weight, "
        "Tensor inverse_rms, Tensor grad_output, float epsilon) "
        "-> (Tensor dx, Tensor dgate, Tensor dweight)");
}

TORCH_LIBRARY_IMPL(nanochat_kda, CUDA, m) {
  m.impl("rmsnorm_gate_forward", &rmsnorm_gate_forward_cuda);
  m.impl("rmsnorm_gate_backward", &rmsnorm_gate_backward_cuda);
}
