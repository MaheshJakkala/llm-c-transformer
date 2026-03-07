// ffn.h
#ifndef FFN_H
#define FFN_H

#include "tensor.h"
#include "linear.h"
#include "activations.h"
#include "ops.h"
#include <string.h>
#include "qtensor.h"


// Feedforward forward:
// input: (seq x d_model)
// W1: d_model x hidden, b1: 1 x hidden
// W2: hidden x d_model, b2: 1 x d_model
// out: seq x d_model
// temp1: seq x hidden
void ffn_forward(const Tensor *input,
                 const Tensor *W1, const Tensor *b1,
                 const Tensor *W2, const Tensor *b2,
                 Tensor *out, Tensor *temp1);

void ffn_backward(
    const Tensor *res1,
    const Tensor *ff1,
    const Tensor *grad_out,
    const Tensor *W2,
    const Tensor *W1,          // ✅ ADD THIS
    Tensor *grad_W2,
    Tensor *grad_b2,
    Tensor *grad_W1,
    Tensor *grad_b1,
    Tensor *grad_res1
);

void ffn_forward_q8(
    const QTensor *input_q,
    const QTensor *W1_q,
    const Tensor  *b1,
    const QTensor *W2_q,
    const Tensor  *b2,
    Tensor *out,
    Tensor *temp1
);

#endif
