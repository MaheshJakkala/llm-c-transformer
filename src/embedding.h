// src/embedding.h
#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "tensor.h"

typedef struct {
    Tensor *weights;  // vocab_size x hidden_size
} Embedding;

Embedding *embedding_create(int vocab_size, int hidden_size);
void embedding_free(Embedding *e);
void embedding_forward(Embedding *e, int *input_ids, int seq_len, Tensor *out);
void embedding_backward(
    Embedding *e,
    int *input_ids,
    int seq_len,
    Tensor *grad_x,
    Tensor *grad_emb
);

#endif
