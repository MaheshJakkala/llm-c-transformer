// transformer_block.c
#include "transformer_block.h"

void transformer_block_forward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
    const Tensor *W1, const Tensor *b1,
    const Tensor *W2, const Tensor *b2,
    Tensor *out,
    Tensor **ff1,
    AttentionCache *cache, const int *padding_mask
){
    int seq = x->rows;
    int hidden = x->cols;

    // Allocate FFN intermediate once
    if (*ff1 == NULL) {
        *ff1 = tensor_create(seq, W1->cols);
    }
    if (!out) out = tensor_create(seq, hidden);

    // Allocate attention output and STORE in cache
    Tensor *attn_out = tensor_create(seq, hidden);
    
    attention_forward(x, Wq, Wk, Wv, Wo,padding_mask, attn_out, cache);
    // Tensor *attn_out = cache->attn_out;
    // residual 1: attn + x
    for (int i = 0; i < seq * hidden; i++)
        attn_out->data[i] += x->data[i];
    layernorm(attn_out, 1e-5);

    if (!(*ff1)) {
    fprintf(stderr, "FATAL: ff1 is NULL before FFN\n");
    exit(1);
}
    // FFNhidden = 256
   
    linear_forward(cache->attn_out, W1, b1, *ff1);
    gelu_inplace(*ff1);

    if (!(*ff1)) {
    fprintf(stderr, "FATAL: ff1 is NULL before FFN\n");
    exit(1);
}

    linear_forward(*ff1, W2, b2, out);

    // residual 2: ffn + attn_outhidden = 256
    for (int i = 0; i < seq * hidden; i++)
        out->data[i] += attn_out->data[i];
    layernorm(out, 1e-5);
}

// #include "transformer_block.h"

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
) {
    int seq = x_q->rows;
    int hidden = x_q->cols;

    if (*ff1 == NULL)
        *ff1 = tensor_create(seq, W1_q->cols);

    Tensor *attn_out = tensor_create(seq, hidden);

    attention_forward_q8(
        x_q,
        Wq_q, Wk_q, Wv_q, Wo_q,
        padding_mask,
        attn_out,
        cache
    );

    layernorm(attn_out, 1e-5);

    QTensor *attn_q = quantize_tensor(attn_out);

    ffn_forward_q8(
        attn_q,
        W1_q, b1,
        W2_q, b2,
        out,
        *ff1
    );

    for (int i = 0; i < seq * hidden; i++)
        out->data[i] += attn_out->data[i];

    layernorm(out, 1e-5);

    free_qtensor(attn_q);
    tensor_free(attn_out);
}
