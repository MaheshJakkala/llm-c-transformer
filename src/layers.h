#ifndef LAYERS_H
#define LAYERS_H

#include "tensor.h"
// y = xW + b
// void linear_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *out);

// void linear_backward(
//     const Tensor *x,
//     const Tensor *W,
//     const Tensor *grad_out,
//     Tensor *grad_x,
//     Tensor *grad_W,
//     Tensor *grad_b
// );

// ReLU: in-place
void relu_forward(Tensor *x);

// GELU: out-of-place
void gelu_forward(const Tensor *x, Tensor *y);

// Numerically stable softmax along last dimension
void softmax_forward(const Tensor *x, Tensor *y);

// LayerNorm: normalize across last dimension
void layernorm_forward(const Tensor *x, const Tensor *gamma, const Tensor *beta, Tensor *y, float eps);

#endif