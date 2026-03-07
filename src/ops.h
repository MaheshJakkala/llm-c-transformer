// ops.h
#ifndef OPS_H
#define OPS_H

#include "tensor.h"

// C = A x B  (A: MxK, B: KxN, C: MxN)
void matmul_naive(const Tensor *A, const Tensor *B, Tensor *C);

// C = A x B^T  (A: MxK, B: N x K, C: M x N)
void matmul_transposeB(const Tensor *A, const Tensor *B, Tensor *C);
// void matmul_naive(Tensor *A, Tensor *B, Tensor *C);

// // y = xW + b
// void linear_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *y);

// // ReLU: in-place
// void relu_forward(Tensor *x);

// // GELU: out-of-place
// void gelu_forward(const Tensor *x, Tensor *y);

// // Numerically stable softmax along last dimension
// void softmax_forward(const Tensor *x, Tensor *y);

// // LayerNorm: normalize across last dimension
// void layernorm_forward(const Tensor *x, const Tensor *gamma, const Tensor *beta, Tensor *y, float eps);


#endif
