// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"
#include <math.h>
#include <stdio.h>


void gelu_forward(const Tensor *in, Tensor *out);
void gelu_backward(const Tensor *x, const Tensor *grad_out, Tensor *grad_in);

// softmax per row: in -> out (numerically stable)
void softmax_rows(const Tensor *in, Tensor *out);
// In-place GELU: out = GELU(x)
void gelu_inplace(Tensor *t);


#endif
