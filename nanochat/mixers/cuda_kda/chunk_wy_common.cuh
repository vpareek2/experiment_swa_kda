// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Andrej Karpathy
//
// Independent implementation of the published KDA WY/UT equations.
#pragma once

#include <ATen/ATen.h>

at::Tensor nanochat_kda_chunk_wy_forward_c64(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& raw_gate,
    const at::Tensor& beta_logits,
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    float lower_bound,
    float scale);
