// attention.c
#include "attention.h"

void attention_cache_free(AttentionCache *cache) {
    if (!cache) return;
    tensor_free(cache->Q);
    tensor_free(cache->K);
    tensor_free(cache->V);
    tensor_free(cache->scores);
    tensor_free(cache->attn);
    tensor_free(cache->attn_out);

    // memset(cache, 0, sizeof(*cache));
}

void attention_core(
    const Tensor *Q,
    const Tensor *K,
    const Tensor *V,
    const int *padding_mask,
    Tensor *out,
    AttentionCache *cache
) {
    int seq = Q->rows;
    int hidden = Q->cols;

    /* Allocate cache tensors if needed */
    if (!cache->scores)    cache->scores    = tensor_create(seq, seq);
    if (!cache->attn)      cache->attn      = tensor_create(seq, seq);
    if (!cache->attn_out)  cache->attn_out  = tensor_create(seq, hidden);

    float scale = 1.0f / sqrtf((float)hidden);

    /* scores = Q K^T / sqrt(d) */
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                sum += Q->data[i*hidden + k] *
                       K->data[j*hidden + k];
            }
            cache->scores->data[i*seq + j] = sum * scale;
        }
    }

    /* Padding mask */
    if (padding_mask) {
        for (int i = 0; i < seq; ++i) {
            for (int j = 0; j < seq; ++j) {
                if (padding_mask[j] == 0) {
                    cache->scores->data[i*seq + j] = -1e9f;
                }
            }
        }
    }

    /* Softmax */
    for (int i = 0; i < seq; ++i) {
        float max_val = -INFINITY;

        for (int j = 0; j < seq; ++j) {
            float v = cache->scores->data[i*seq + j];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float e = expf(cache->scores->data[i*seq + j] - max_val);
            cache->attn->data[i*seq + j] = e;
            sum += e;
        }

        for (int j = 0; j < seq; ++j) {
            cache->attn->data[i*seq + j] /= sum;
        }
    }

    /* attn_out = attn * V */
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < hidden; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq; ++k) {
                sum += cache->attn->data[i*seq + k] *
                       V->data[k*hidden + j];
            }
            cache->attn_out->data[i*hidden + j] = sum;
        }
    }

    /* Copy to out (before Wo projection in float path) */
    for (int i = 0; i < seq * hidden; ++i) {
        out->data[i] = cache->attn_out->data[i];
    }
}

// ---------------- Forward Pass ----------------
void attention_forward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
     const int *padding_mask,
    Tensor *out,
    AttentionCache *cache
) {
    int seq = x->rows;
    int hidden = x->cols;

    // Allocate cache tensors ONCE
    if (!cache->Q)         cache->Q         = tensor_create(seq, hidden);
    if (!cache->K)         cache->K         = tensor_create(seq, hidden);
    if (!cache->V)         cache->V         = tensor_create(seq, hidden);
    if (!cache->scores)    cache->scores    = tensor_create(seq, seq);   // pre-softmax
    if (!cache->attn)      cache->attn      = tensor_create(seq, seq);   // post-softmax
    if (!cache->attn_out)  cache->attn_out  = tensor_create(seq, hidden);

    if (!cache->Q || !cache->K || !cache->V ||
        !cache->scores || !cache->attn || !cache->attn_out) {
        fprintf(stderr, "attention_forward: allocation failed\n");
        exit(1);
    }

    // Q, K, V projections
    linear_forward(x, Wq, NULL, cache->Q);
    
    linear_forward(x, Wk, NULL, cache->K);
    linear_forward(x, Wv, NULL, cache->V);

    float scale = 1.0f / sqrtf((float)hidden);

    // scores = Q K^T / sqrt(d)
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                sum += cache->Q->data[i*hidden + k] *
                       cache->K->data[j*hidden + k];
            }
            cache->scores->data[i*seq + j] = sum * scale;
        }
    }

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            if (padding_mask[j] == 0) {
                cache->scores->data[i*seq + j] = -1e9f; // mask PAD keys
            }
        }
    }

    // softmax(scores -> attn)
    for (int i = 0; i < seq; ++i) {
        float max_val = -INFINITY;
        for (int j = 0; j < seq; ++j) {
            float v = cache->scores->data[i*seq + j];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float e = expf(cache->scores->data[i*seq + j] - max_val);
            cache->attn->data[i*seq + j] = e;
            sum += e;
        }

        for (int j = 0; j < seq; ++j)
            cache->attn->data[i*seq + j] /= sum;
    }

    // attn_out = attn * V
    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < hidden; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < seq; ++k) {
                sum += cache->attn->data[i*seq + k] *
                       cache->V->data[k*hidden + j];
            }
            cache->attn_out->data[i*hidden + j] = sum;
        }
    }

    
    // final projection
    linear_forward(cache->attn_out, Wo, NULL, out);
}

// ---------------- Backward Pass ----------------
void attention_backward(
    const Tensor *x,
    const Tensor *Wq, const Tensor *Wk, const Tensor *Wv, const Tensor *Wo,
    const Tensor *grad_out,
    const AttentionCache *cache,
    Tensor *grad_x,
    Tensor *grad_Wq, Tensor *grad_Wk, Tensor *grad_Wv, Tensor *grad_Wo
) {
    // printf("entered attention_backward\n"); fflush(stdout);
    CHECK(cache->Q);
    CHECK(cache->K);
    CHECK(cache->V);
    CHECK(cache->scores);
    CHECK(cache->attn);
    CHECK(cache->attn_out);

    int N = x->rows;
    int d = x->cols;

    ASSERT_SHAPE(cache->Q, N, d);
    ASSERT_SHAPE(cache->K, N, d);
    ASSERT_SHAPE(cache->V, N, d);
    ASSERT_SHAPE(cache->scores, N, N);
    ASSERT_SHAPE(cache->attn, N, N);
    ASSERT_SHAPE(cache->attn_out, N, d);
    ASSERT_SHAPE(grad_out, N, d);

    int seq = x->rows;
    int hidden = x->cols;

    // 1. Backprop through final projection
    // Tensor *grad_attn_out = tensor_create(seq, hidden);
    // grad_out IS grad_attn_out
// grad_attn_out must be a NEW tensor
    Tensor *grad_attn_out = tensor_create(seq, hidden);
    tensor_zero(grad_attn_out);

    linear_backward(
        cache->attn_out,
        Wo,
        grad_out,
        grad_attn_out,
        grad_Wo,
        NULL
    );

    // 2. Gradients w.r.t V and attn
    Tensor *grad_V = tensor_create(seq, hidden);
    Tensor *grad_attn = tensor_create(seq, seq);
    tensor_zero(grad_V);
    tensor_zero(grad_attn);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < hidden; ++k) {
                grad_V->data[j*hidden + k] +=
                    grad_attn_out->data[i*hidden + k] *
                    cache->attn->data[i*seq + j];

                sum += grad_attn_out->data[i*hidden + k] *
                       cache->V->data[j*hidden + k];
            }
            grad_attn->data[i*seq + j] = sum;
        }
    }

    // 3. Softmax backward (attn -> scores)
    Tensor *grad_scores = tensor_create(seq, seq);
    for (int i = 0; i < seq; ++i) {
        float row_sum = 0.0f;
        for (int j = 0; j < seq; ++j)
            row_sum += grad_attn->data[i*seq + j] *
                       cache->attn->data[i*seq + j];

        for (int j = 0; j < seq; ++j) {
            grad_scores->data[i*seq + j] =
                cache->attn->data[i*seq + j] *
                (grad_attn->data[i*seq + j] - row_sum);
            // grad_scores->data[i*seq + j] *= 1/sqrt(hidden);

        }
    }

    // 4. Gradients w.r.t Q and K
    Tensor *grad_Q = tensor_create(seq, hidden);
    Tensor *grad_K = tensor_create(seq, hidden);
    tensor_zero(grad_Q);
    tensor_zero(grad_K);

    float scale = 1.0f / sqrtf((float)hidden);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            for (int k = 0; k < hidden; ++k) {
                grad_Q->data[i*hidden + k] +=
                    grad_scores->data[i*seq + j] *
                    cache->K->data[j*hidden + k] * scale;

                grad_K->data[j*hidden + k] +=
                    grad_scores->data[i*seq + j] *
                    cache->Q->data[i*hidden + k] * scale;
            }
        }
    }

    // 5. Backprop through Q, K, V linear layers
    tensor_zero(grad_x);
    // printf("\ngrad_x in attention_backward(): ");
    // tensor_print(grad_x);

    linear_backward(x, Wq, grad_Q, grad_x, grad_Wq, NULL);

    Tensor *grad_x_tmp = tensor_create(seq, hidden);
    linear_backward(x, Wk, grad_K, grad_x_tmp, grad_Wk, NULL);
    for (int i = 0; i < seq * hidden; ++i)
        grad_x->data[i] += grad_x_tmp->data[i];
    tensor_free(grad_x_tmp);

    grad_x_tmp = tensor_create(seq, hidden);
    linear_backward(x, Wv, grad_V, grad_x_tmp, grad_Wv, NULL);
    for (int i = 0; i < seq * hidden; ++i)
        grad_x->data[i] += grad_x_tmp->data[i];
    tensor_free(grad_x_tmp);

//     printf("\nfinal grad_x: ");
// tensor_print(grad_x);

    // Cleanup
    // tensor_free(grad_attn_out);
    tensor_free(grad_V);
    tensor_free(grad_attn);
    tensor_free(grad_scores);
    tensor_free(grad_Q);
    tensor_free(grad_K);
    tensor_free(grad_attn_out);
}

// #include "attention.h"

// void attention_forward_q8(
//     const QTensor *x_q,
//     const QTensor *Wq_q,
//     const QTensor *Wk_q,
//     const QTensor *Wv_q,
//     const QTensor *Wo_q,
//     const int *padding_mask,
//     Tensor *out,
//     AttentionCache *cache
// ) {
//     int seq = x_q->rows;
//     int hidden = x_q->cols;

//     Tensor *Q = tensor_create(seq, hidden);
//     Tensor *K = tensor_create(seq, hidden);
//     Tensor *V = tensor_create(seq, hidden);

//     linear_forward_q8(x_q, Wq_q, NULL, Q);
//     linear_forward_q8(x_q, Wk_q, NULL, K);
//     linear_forward_q8(x_q, Wv_q, NULL, V);

//     /* reuse your EXISTING float attention logic */
//     attention_forward(Q, K, V, NULL, padding_mask, out, cache);

//     QTensor *out_q = quantize_tensor(out);
//     linear_forward_q8(out_q, Wo_q, NULL, out);

//     free_qtensor(out_q);
//     tensor_free(Q); tensor_free(K); tensor_free(V);
// }

void attention_forward_q8(
    const QTensor *x_q,
    const QTensor *Wq_q,
    const QTensor *Wk_q,
    const QTensor *Wv_q,
    const QTensor *Wo_q,
    const int *padding_mask,
    Tensor *out,
    AttentionCache *cache
) {
    int seq = x_q->rows;
    int hidden = x_q->cols;

    /* Dequantize projections */
    Tensor *Q = tensor_create(seq, hidden);
    Tensor *K = tensor_create(seq, hidden);
    Tensor *V = tensor_create(seq, hidden);
    // printf("x_q: (%d,%d)\n", x_q->rows, x_q->cols);
// printf("W_q: (%d,%d)\n", Wq_q->rows, Wq_q->cols);
    // printf("hello before in q8");
    linear_forward_q8(x_q, Wq_q, NULL, Q);
    // printf("hello before in q8");
    linear_forward_q8(x_q, Wk_q, NULL, K);
    linear_forward_q8(x_q, Wv_q, NULL, V);

    /* Use EXISTING FLOAT attention */
    attention_core(Q, K, V, padding_mask, out, cache);


    /* Final projection in quantized form */
    QTensor *out_q = quantize_tensor(out);
    // printf("hello before in q8");
    // QTensor *out_q = quantize_weight_transpose(out);

    linear_forward_q8(out_q, Wo_q, NULL, out);
    //  printf("hello after in q8");

    free_qtensor(out_q);
    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
}


// void attention_forward_q8(
//     const QTensor *x_q,
//     const QTensor *Wq_q,
//     const QTensor *Wk_q,
//     const QTensor *Wv_q,
//     const QTensor *Wo_q,
//     const int *padding_mask,
//     Tensor *out,
//     AttentionCache *cache
// ) {
//     int seq = x_q->rows;
//     int hidden = x_q->cols;

//     /* Dequantized Q,K,V */
//     Tensor *Q = tensor_create(seq, hidden);
//     Tensor *K = tensor_create(seq, hidden);
//     Tensor *V = tensor_create(seq, hidden);

//     linear_forward_q8(x_q, Wq_q, NULL, Q);
//     linear_forward_q8(x_q, Wk_q, NULL, K);
//     linear_forward_q8(x_q, Wv_q, NULL, V);

//     /* Core attention (float) */
//     // attention_core(Q, K, V, padding_mask, out, cache);
//     int32_t *scores_int = malloc(seq * seq * sizeof(int32_t));

// matmul_q8_q8_int32(Q_q, K_q, scores_int);

// float scale = (Q_q->scale * K_q->scale) / sqrtf((float)hidden);

// for (int i = 0; i < seq * seq; ++i) {
//     cache->scores->data[i] = scores_int[i] * scale;
// }


//     /* Quantized output projection */
//     QTensor *out_q = quantize_tensor(out);
//     linear_forward_q8(out_q, Wo_q, NULL, out);

//     free_qtensor(out_q);
//     tensor_free(Q);
//     tensor_free(K);
//     tensor_free(V);
// }


