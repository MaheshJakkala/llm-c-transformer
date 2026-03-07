// src/attention.h
#ifndef ATTENTION_H
#define CHECK(t) if (!(t)) { \
    printf("NULL tensor: %s\n", #t); fflush(stdout); abort(); \
}
#define ASSERT_SHAPE(t, r, c) \
    if ((t)->rows != (r) || (t)->cols != (c)) { \
        printf("Shape mismatch %s: got %dx%d expected %dx%d\n", \
            #t, (t)->rows, (t)->cols, (r), (c)); \
        abort(); \
    }

#define ATTENTION_H



#include "qtensor.h"
#include "tensor.h"
#include "linear.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    Tensor *Q;        // (N x d)
    Tensor *K;        // (N x d)
    Tensor *V;        // (N x d)
    Tensor *scores;   // (N x N) BEFORE softmax
    Tensor *attn;     // (N x N) AFTER softmax
    Tensor *attn_out; // (N x d)
} AttentionCache;

void attention_core(
    const Tensor *Q,
    const Tensor *K,
    const Tensor *V,
    const int *padding_mask,
    Tensor *out,
    AttentionCache *cache
);

void attention_cache_free(AttentionCache *cache);
// Full self-attention: computes Q,K,V from x with learned weights
// Forward pass
void attention_forward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
    const int *padding_mask,  
    Tensor *out,
    AttentionCache *cache // optional, can be NULL
);

// Backward pass
void attention_backward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
    const Tensor *grad_out,
    const AttentionCache *cache, // store forward intermediates
    Tensor *grad_x,              // gradient w.r.t input
    Tensor *grad_Wq, Tensor *grad_Wk, Tensor *grad_Wv, Tensor *grad_Wo
);

void attention_forward_q8(
    const QTensor *x_q,
    const QTensor *Wq_q,
    const QTensor *Wk_q,
    const QTensor *Wv_q,
    const QTensor *Wo_q,
    const int *padding_mask,
    Tensor *out,
    AttentionCache *cache
);

#endif
