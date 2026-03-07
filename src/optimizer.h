#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

typedef struct {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    Tensor *m; // first moment
    Tensor *v; // second moment
    int t;     // timestep
} AdamState;

AdamState* adam_create(float lr, float beta1, float beta2, float epsilon, int rows, int cols);
void adam_step(Tensor *param, const Tensor *grad, AdamState *state);
void adam_free(AdamState *state);

#endif
