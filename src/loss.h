#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

// Compute cross-entropy loss for logits and integer targets
// loss.h
float cross_entropy_loss(const Tensor *pred, const int *target);
void cross_entropy_loss_grad(const Tensor *pred, const int *target, Tensor *grad);

#endif
