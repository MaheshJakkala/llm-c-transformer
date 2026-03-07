#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "tensor.h"

void layernorm_forward(const Tensor *x, const Tensor *gamma, const Tensor *beta, Tensor *y, float eps);
void layernorm(Tensor *x, float eps);

#endif
