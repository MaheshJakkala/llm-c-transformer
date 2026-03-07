// linear.h
#ifndef LINEAR_H
#define LINEAR_H

#include "tensor.h"
#include "qtensor.h"

// out = X * W + b
// X: (batch*seq) x in_dim, W: in_dim x out_dim, b: 1 x out_dim (or NULL)
void linear_forward(const Tensor *X, const Tensor *W, const Tensor *b, Tensor *out);
void linear_backward(
    const Tensor *x,
    const Tensor *W,
    const Tensor *grad_out,
    Tensor *grad_x,
    Tensor *grad_W,
    Tensor *grad_b
);

void linear_forward_q8(
    const QTensor *input,
    const QTensor *weight,
    const Tensor *bias,
    Tensor *out
);

#endif
