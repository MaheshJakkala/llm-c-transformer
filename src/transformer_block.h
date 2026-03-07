// transformer_block.h
#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "tensor.h"
#include "ffn.h"
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "linear.h"
#include "activations.h"
#include "layernorm.h"
#include "attention.h"
#include "qtensor.h"

void transformer_block_forward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *out,
    Tensor **ff1,
    AttentionCache *cache, const int *padding_mask
);

void transformer_block_forward_q8(
    const QTensor *x_q,
    const QTensor *Wq_q,
    const QTensor *Wk_q,
    const QTensor *Wv_q,
    const QTensor *Wo_q,
    const QTensor *W1_q,
    const Tensor  *b1,
    const QTensor *W2_q,
    const Tensor  *b2,
    Tensor *out,
    Tensor **ff1,
    AttentionCache *cache,
    const int *padding_mask
);

#endif
