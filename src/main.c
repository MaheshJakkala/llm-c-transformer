/* main.c – Transformer NER training on CoNLL-2003 */
#include "main.h"
#include <string.h>
#include <time.h>

/* ─── Gradient clipping (L2 norm) ─────────────────────────────────────── */
static void clip_grad(Tensor *g, float max_norm) {
    if (!g) return;
    int n = g->rows * g->cols;
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += g->data[i] * g->data[i];
    norm = sqrtf(norm);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (int i = 0; i < n; i++) g->data[i] *= scale;
    }
}

/* ─── Training ─────────────────────────────────────────────────────────── */
void train(
    char train_texts[TRAIN_SAMPLES][128],
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
    char val_texts[VAL_SAMPLES][128],
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
) {
    int hidden      = HIDDEN_SIZE;
    int ffn_hidden  = FFN_HIDDEN;
    int num_classes = NUM_CLASSES;

    /* ── Tokenizer & Embedding ── */
    Tokenizer *tok = tokenizer_create();
    Embedding *emb = embedding_create(tok->vocab_size, hidden);
    Tensor    *grad_emb = tensor_create(tok->vocab_size, hidden);
    tensor_zero(grad_emb);

    /* ── Weights ── */
    Tensor *Wq = tensor_create(hidden, hidden); tensor_fill_random(Wq, -0.02f, 0.02f);
    Tensor *Wk = tensor_create(hidden, hidden); tensor_fill_random(Wk, -0.02f, 0.02f);
    Tensor *Wv = tensor_create(hidden, hidden); tensor_fill_random(Wv, -0.02f, 0.02f);
    Tensor *Wo = tensor_create(hidden, hidden); tensor_fill_random(Wo, -0.02f, 0.02f);

    Tensor *W1 = tensor_create(hidden, ffn_hidden);  tensor_fill_random(W1, -0.02f, 0.02f);
    Tensor *b1 = tensor_create(1,      ffn_hidden);  tensor_zero(b1);
    Tensor *W2 = tensor_create(ffn_hidden, hidden);  tensor_fill_random(W2, -0.02f, 0.02f);
    Tensor *b2 = tensor_create(1,      hidden);      tensor_zero(b2);

    Tensor *W_cls = tensor_create(hidden, num_classes); tensor_fill_random(W_cls, -0.02f, 0.02f);
    Tensor *b_cls = tensor_create(1, num_classes);      tensor_zero(b_cls);

    /* ── Adam states ── */
    AdamState *opt_Wq    = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      hidden);
    AdamState *opt_Wk    = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      hidden);
    AdamState *opt_Wv    = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      hidden);
    AdamState *opt_Wo    = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      hidden);
    AdamState *opt_W1    = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      ffn_hidden);
    AdamState *opt_b1    = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1,           ffn_hidden);
    AdamState *opt_W2    = adam_create(LR, 0.9f, 0.999f, 1e-8f, ffn_hidden,  hidden);
    AdamState *opt_b2    = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1,           hidden);
    AdamState *opt_Wcls  = adam_create(LR, 0.9f, 0.999f, 1e-8f, hidden,      num_classes);
    AdamState *opt_bcls  = adam_create(LR, 0.9f, 0.999f, 1e-8f, 1,           num_classes);
    AdamState *opt_emb   = adam_create(LR, 0.9f, 0.999f, 1e-8f, tok->vocab_size, hidden);

    /* ── Persistent gradient buffers ── */
    Tensor *grad_Wq   = tensor_create(hidden, hidden);
    Tensor *grad_Wk   = tensor_create(hidden, hidden);
    Tensor *grad_Wv   = tensor_create(hidden, hidden);
    Tensor *grad_Wo   = tensor_create(hidden, hidden);
    Tensor *grad_W1   = tensor_create(hidden,      ffn_hidden);
    Tensor *grad_b1   = tensor_create(1,            ffn_hidden);
    Tensor *grad_W2   = tensor_create(ffn_hidden,  hidden);
    Tensor *grad_b2   = tensor_create(1,            hidden);
    Tensor *grad_Wcls = tensor_create(hidden, num_classes);
    Tensor *grad_bcls = tensor_create(1,      num_classes);

    printf("Training Transformer NER (hidden=%d, ffn=%d, vocab=%d, epochs=%d)\n",
           hidden, ffn_hidden, tok->vocab_size, EPOCHS);
    printf("-------------------------------------------------------------------\n");

    struct timespec t_train_start, t_train_end;
    clock_gettime(CLOCK_MONOTONIC, &t_train_start);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;

        for (int s = 0; s < TRAIN_SAMPLES; ++s) {
            int input_ids[MAX_SEQ_LEN] = {0};
            int seq_len;
            encode_word(train_texts[s], input_ids, MAX_SEQ_LEN, &seq_len);
            if (seq_len > MAX_SEQ_LEN) seq_len = MAX_SEQ_LEN;

            int padding_mask[MAX_SEQ_LEN] = {0};
            for (int i = 0; i < seq_len; i++) padding_mask[i] = 1;

            /* ─ Forward ─ */
            Tensor *x        = tensor_create(seq_len, hidden);
            Tensor *out      = tensor_create(seq_len, hidden);
            Tensor *logits   = tensor_create(seq_len, num_classes);
            Tensor *grad_logits = tensor_create(seq_len, num_classes);
            Tensor *grad_out = tensor_create(seq_len, hidden);
            Tensor *grad_x   = tensor_create(seq_len, hidden);
            Tensor *ff1      = NULL;
            AttentionCache cache = {0};

            embedding_forward(emb, input_ids, seq_len, x);

            /* Sinusoidal-like positional signal */
            for (int i = 0; i < seq_len; i++)
                x->data[i * hidden + (i % hidden)] += 0.01f;

            transformer_block_forward(
                x, Wq, Wk, Wv, Wo,
                W1, b1, W2, b2,
                out, &ff1, &cache, padding_mask
            );

            linear_forward(out, W_cls, b_cls, logits);

            /* Build target array */
            int target[MAX_SEQ_LEN];
            for (int i = 0; i < seq_len; i++) {
                int lbl = train_labels[s][i];
                target[i] = (lbl >= 0 && lbl < num_classes) ? lbl : -1;
            }
            for (int i = seq_len; i < MAX_SEQ_LEN; i++) target[i] = -1;

            float loss = cross_entropy_loss(logits, target);
            epoch_loss += loss;
            cross_entropy_loss_grad(logits, target, grad_logits);

            /* ─ Backward ─ */
            tensor_zero(grad_Wcls); tensor_zero(grad_bcls);
            linear_backward(out, W_cls, grad_logits, grad_out, grad_Wcls, grad_bcls);

            tensor_zero(grad_x);
            tensor_zero(grad_Wq); tensor_zero(grad_Wk);
            tensor_zero(grad_Wv); tensor_zero(grad_Wo);
            tensor_zero(grad_W1); tensor_zero(grad_b1);
            tensor_zero(grad_W2); tensor_zero(grad_b2);

            Tensor *grad_attn_out = tensor_create(seq_len, hidden);

            ffn_backward(
                cache.attn_out, ff1, grad_out,
                W2, W1,
                grad_W2, grad_b2, grad_W1, grad_b1,
                grad_attn_out
            );

            attention_backward(
                x, Wq, Wk, Wv, Wo,
                grad_attn_out, &cache,
                grad_x, grad_Wq, grad_Wk, grad_Wv, grad_Wo
            );

            tensor_free(grad_attn_out);

            tensor_zero(grad_emb);
            embedding_backward(emb, input_ids, seq_len, grad_x, grad_emb);

            /* ─ Gradient clipping ─ */
            clip_grad(grad_Wq,   GRAD_CLIP); clip_grad(grad_Wk,   GRAD_CLIP);
            clip_grad(grad_Wv,   GRAD_CLIP); clip_grad(grad_Wo,   GRAD_CLIP);
            clip_grad(grad_W1,   GRAD_CLIP); clip_grad(grad_b1,   GRAD_CLIP);
            clip_grad(grad_W2,   GRAD_CLIP); clip_grad(grad_b2,   GRAD_CLIP);
            clip_grad(grad_Wcls, GRAD_CLIP); clip_grad(grad_bcls, GRAD_CLIP);
            clip_grad(grad_emb,  GRAD_CLIP);

            /* ─ Adam updates ─ */
            adam_step(Wq,          grad_Wq,   opt_Wq);
            adam_step(Wk,          grad_Wk,   opt_Wk);
            adam_step(Wv,          grad_Wv,   opt_Wv);
            adam_step(Wo,          grad_Wo,   opt_Wo);
            adam_step(W1,          grad_W1,   opt_W1);
            adam_step(b1,          grad_b1,   opt_b1);
            adam_step(W2,          grad_W2,   opt_W2);
            adam_step(b2,          grad_b2,   opt_b2);
            adam_step(W_cls,       grad_Wcls, opt_Wcls);
            adam_step(b_cls,       grad_bcls, opt_bcls);
            adam_step(emb->weights,grad_emb,  opt_emb);

            /* ─ Cleanup per-sample ─ */
            tensor_free(cache.Q);     tensor_free(cache.K);     tensor_free(cache.V);
            tensor_free(cache.scores);tensor_free(cache.attn);  tensor_free(cache.attn_out);
            tensor_free(x); tensor_free(out); tensor_free(logits);
            tensor_free(grad_logits); tensor_free(grad_out); tensor_free(grad_x);
            if (ff1) tensor_free(ff1);
        }

        printf("Epoch %3d/%d | Loss = %.4f\n", epoch + 1, EPOCHS,
               epoch_loss / TRAIN_SAMPLES);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_train_end);
    double train_sec = (t_train_end.tv_sec  - t_train_start.tv_sec) +
                       (t_train_end.tv_nsec - t_train_start.tv_nsec) * 1e-9;
    printf("\nTraining wall-time: %.2f s (%.1f ms/epoch)\n",
           train_sec, 1000.0 * train_sec / EPOCHS);

    /* ── INT8 quantized evaluation ── */
    printf("\n============================================================\n");
    printf(" INT8 Quantized Inference (Post-Training)\n");
    printf("============================================================\n");

    char *val_text_ptrs[VAL_SAMPLES];
    int  *val_label_ptrs[VAL_SAMPLES];
    for (int i = 0; i < VAL_SAMPLES; i++) {
        val_text_ptrs[i]  = val_texts[i];
        val_label_ptrs[i] = val_labels[i];
    }

    test_model_q8(
        emb, tok,
        Wq, Wk, Wv, Wo,
        W1, b1, W2, b2,
        W_cls, b_cls,
        val_text_ptrs, val_label_ptrs, VAL_SAMPLES
    );

    /* ── Memory footprint ── */
    printf("\n============================================================\n");
    printf(" Memory Footprint\n");
    printf("============================================================\n");
    memoryfootprint(emb);

    /* ── Cleanup ── */
    tensor_free(Wq); tensor_free(Wk); tensor_free(Wv); tensor_free(Wo);
    tensor_free(W1); tensor_free(b1); tensor_free(W2); tensor_free(b2);
    tensor_free(W_cls); tensor_free(b_cls);
    tensor_free(grad_Wq); tensor_free(grad_Wk); tensor_free(grad_Wv); tensor_free(grad_Wo);
    tensor_free(grad_W1); tensor_free(grad_b1); tensor_free(grad_W2); tensor_free(grad_b2);
    tensor_free(grad_Wcls); tensor_free(grad_bcls); tensor_free(grad_emb);
    adam_free(opt_Wq); adam_free(opt_Wk); adam_free(opt_Wv); adam_free(opt_Wo);
    adam_free(opt_W1); adam_free(opt_b1); adam_free(opt_W2); adam_free(opt_b2);
    adam_free(opt_Wcls); adam_free(opt_bcls); adam_free(opt_emb);
    embedding_free(emb);
    tokenizer_free(tok);
}

/* ─── main ─────────────────────────────────────────────────────────────── */
int main(void) {
    srand(42);

    char train_texts[TRAIN_SAMPLES][128];
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN];
    char val_texts[VAL_SAMPLES][128];
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN];

    int train_n = 0, val_n = 0;

    load_conll_dataset("data/conll2003/train.json",
                       train_texts, train_labels, TRAIN_SAMPLES, &train_n);
    load_conll_dataset("data/conll2003/valid.json",
                       val_texts, val_labels, VAL_SAMPLES, &val_n);

    printf("Loaded %d train, %d val samples\n", train_n, val_n);
    train(train_texts, train_labels, val_texts, val_labels);
    return 0;
}
