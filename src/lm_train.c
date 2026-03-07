/*
 * lm_train.c — Causal Character-Level Language Model
 *
 * This IS the LLM component of the project.
 *
 * Task: next-token prediction on raw text (corpus.txt).
 *       Given 32 characters of context, predict the 33rd.
 *       Loss = cross-entropy over 256-class vocabulary.
 *       This is exactly what GPT-style language models are trained on,
 *       at character level instead of sub-word BPE level.
 *
 * Architecture:
 *   bytes → Embedding(256 × 128) → TransformerBlock(causal mask)
 *        → Linear(128 → 256) → softmax → next-char prediction
 *
 * Memory-efficient training:
 *   All per-step activation tensors are allocated from a TensorArena
 *   (bump allocator). arena_reset() at the end of each step reuses
 *   the same memory instead of calling malloc/free thousands of times.
 *
 * Compile & run:
 *   make lm
 *   ./lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "lm_config.h"
#include "tensor.h"
#include "qtensor.h"
#include "linear.h"
#include "layernorm.h"
#include "activations.h"
#include "attention.h"
#include "ffn.h"
#include "transformer_block.h"
#include "loss.h"
#include "optimizer.h"
#include "embedding.h"
#include "arena.h"

/* ─── Helpers ────────────────────────────────────────────────────────────── */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec * 1e-6;
}

static void clip_grad(Tensor *g, float max_norm) {
    if (!g) return;
    int n = g->rows * g->cols;
    float norm = 0.0f;
    for (int i = 0; i < n; i++) norm += g->data[i] * g->data[i];
    norm = sqrtf(norm);
    if (norm > max_norm) {
        float s = max_norm / (norm + 1e-6f);
        for (int i = 0; i < n; i++) g->data[i] *= s;
    }
}

/* ─── Causal mask ────────────────────────────────────────────────────────── */
/* padding_mask[j] = 0 blocks position j in attention.
   For a causal LM: query at position i can attend to j ≤ i only. */
static void build_causal_mask(int *mask, int query_pos, int seq) {
    for (int j = 0; j < seq; j++)
        mask[j] = (j <= query_pos) ? 1 : 0;
}

/* ─── Load corpus ────────────────────────────────────────────────────────── */
static unsigned char *load_corpus(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open corpus: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t len = (size_t)ftell(f);
    rewind(f);
    unsigned char *buf = malloc(len + 1);
    fread(buf, 1, len, f);
    buf[len] = '\0';
    fclose(f);
    *len_out = len;
    return buf;
}

/* ─── Sample from logits (temperature sampling) ──────────────────────────── */
static int sample_token(const Tensor *logits, int row, float temperature) {
    int V = logits->cols;
    float probs[256];
    float max_v = logits->data[row * V];
    for (int j = 1; j < V; j++)
        if (logits->data[row * V + j] > max_v)
            max_v = logits->data[row * V + j];

    float sum = 0.0f;
    for (int j = 0; j < V; j++) {
        probs[j] = expf((logits->data[row * V + j] - max_v) / temperature);
        sum += probs[j];
    }
    for (int j = 0; j < V; j++) probs[j] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cumulative = 0.0f;
    for (int j = 0; j < V; j++) {
        cumulative += probs[j];
        if (r <= cumulative) return j;
    }
    return V - 1;
}

/* ─── Model weights ──────────────────────────────────────────────────────── */
typedef struct {
    Embedding *emb;
    Tensor *Wq, *Wk, *Wv, *Wo;
    Tensor *W1, *b1, *W2, *b2;
    Tensor *W_lm;   /* LM head: hidden → vocab */
    Tensor *b_lm;
} LMModel;

typedef struct {
    AdamState *opt_Wq, *opt_Wk, *opt_Wv, *opt_Wo;
    AdamState *opt_W1, *opt_b1, *opt_W2, *opt_b2;
    AdamState *opt_Wlm, *opt_blm;
    AdamState *opt_emb;
} LMOptimizers;

typedef struct {
    Tensor *grad_Wq, *grad_Wk, *grad_Wv, *grad_Wo;
    Tensor *grad_W1, *grad_b1, *grad_W2, *grad_b2;
    Tensor *grad_Wlm, *grad_blm;
    Tensor *grad_emb;
} LMGradients;

static LMModel *model_create(void) {
    LMModel *m = malloc(sizeof(LMModel));
    int H = LM_HIDDEN, F = LM_FFN, V = LM_VOCAB;
    float s = 0.02f;

    m->emb = embedding_create(V, H);

    m->Wq = tensor_create(H, H); tensor_fill_random(m->Wq, -s, s);
    m->Wk = tensor_create(H, H); tensor_fill_random(m->Wk, -s, s);
    m->Wv = tensor_create(H, H); tensor_fill_random(m->Wv, -s, s);
    m->Wo = tensor_create(H, H); tensor_fill_random(m->Wo, -s, s);

    m->W1 = tensor_create(H, F); tensor_fill_random(m->W1, -s, s);
    m->b1 = tensor_create(1, F); tensor_zero(m->b1);
    m->W2 = tensor_create(F, H); tensor_fill_random(m->W2, -s, s);
    m->b2 = tensor_create(1, H); tensor_zero(m->b2);

    m->W_lm = tensor_create(H, V); tensor_fill_random(m->W_lm, -s, s);
    m->b_lm = tensor_create(1, V); tensor_zero(m->b_lm);

    return m;
}

static LMOptimizers *opts_create(LMModel *m) {
    LMOptimizers *o = malloc(sizeof(LMOptimizers));
    int H = LM_HIDDEN, F = LM_FFN, V = LM_VOCAB;
    float lr = LM_LR;
    float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;

    o->opt_Wq  = adam_create(lr, b1, b2, eps, H, H);
    o->opt_Wk  = adam_create(lr, b1, b2, eps, H, H);
    o->opt_Wv  = adam_create(lr, b1, b2, eps, H, H);
    o->opt_Wo  = adam_create(lr, b1, b2, eps, H, H);
    o->opt_W1  = adam_create(lr, b1, b2, eps, H, F);
    o->opt_b1  = adam_create(lr, b1, b2, eps, 1, F);
    o->opt_W2  = adam_create(lr, b1, b2, eps, F, H);
    o->opt_b2  = adam_create(lr, b1, b2, eps, 1, H);
    o->opt_Wlm = adam_create(lr, b1, b2, eps, H, V);
    o->opt_blm = adam_create(lr, b1, b2, eps, 1, V);
    o->opt_emb = adam_create(lr, b1, b2, eps,
                             m->emb->weights->rows, H);
    return o;
}

static LMGradients *grads_create(LMModel *m) {
    LMGradients *g = malloc(sizeof(LMGradients));
    int H = LM_HIDDEN, F = LM_FFN, V = LM_VOCAB;

    g->grad_Wq  = tensor_create(H, H);
    g->grad_Wk  = tensor_create(H, H);
    g->grad_Wv  = tensor_create(H, H);
    g->grad_Wo  = tensor_create(H, H);
    g->grad_W1  = tensor_create(H, F);
    g->grad_b1  = tensor_create(1, F);
    g->grad_W2  = tensor_create(F, H);
    g->grad_b2  = tensor_create(1, H);
    g->grad_Wlm = tensor_create(H, V);
    g->grad_blm = tensor_create(1, V);
    g->grad_emb = tensor_create(m->emb->weights->rows, H);

    return g;
}

/* ─── Text generation ────────────────────────────────────────────────────── */
static void generate(LMModel *m, const char *seed, int gen_len) {
    int H = LM_HIDDEN, V = LM_VOCAB, C = LM_CTX;

    unsigned char ctx[LM_CTX];
    int seed_len = (int)strlen(seed);
    memset(ctx, ' ', C);
    for (int i = 0; i < seed_len && i < C; i++)
        ctx[C - seed_len + i] = (unsigned char)seed[i];

    printf("\n--- Generated text (seed: \"%s\") ---\n%s", seed, seed);

    for (int g = 0; g < gen_len; g++) {
        int ids[LM_CTX];
        for (int i = 0; i < C; i++) ids[i] = ctx[i];

        Tensor *x   = tensor_create(C, H);
        Tensor *out = tensor_create(C, H);
        Tensor *logits = tensor_create(C, V);
        Tensor *ff1    = NULL;
        AttentionCache cache = {0};

        embedding_forward(m->emb, ids, C, x);
        for (int i = 0; i < C; i++)
            x->data[i * H + (i % H)] += 0.01f;

        /* Full causal mask — all positions visible (we use last position's logit) */
        int mask[LM_CTX];
        for (int i = 0; i < C; i++) mask[i] = 1;

        transformer_block_forward(
            x, m->Wq, m->Wk, m->Wv, m->Wo,
            m->W1, m->b1, m->W2, m->b2,
            out, &ff1, &cache, mask
        );
        linear_forward(out, m->W_lm, m->b_lm, logits);

        /* Take the last position's prediction */
        int next = sample_token(logits, C - 1, 0.8f);
        putchar(next);

        /* Slide context window */
        memmove(ctx, ctx + 1, C - 1);
        ctx[C - 1] = (unsigned char)next;

        tensor_free(x); tensor_free(out); tensor_free(logits);
        if (ff1) tensor_free(ff1);
        tensor_free(cache.Q); tensor_free(cache.K); tensor_free(cache.V);
        tensor_free(cache.scores); tensor_free(cache.attn); tensor_free(cache.attn_out);
    }
    printf("\n--- End of generation ---\n\n");
}

/* ─── Main training loop ─────────────────────────────────────────────────── */
int main(void) {
    srand(42);

    printf("============================================================\n");
    printf(" Causal Character-Level Language Model (LLM in C)\n");
    printf(" Vocabulary: %d  Context: %d  Hidden: %d  FFN: %d\n",
           LM_VOCAB, LM_CTX, LM_HIDDEN, LM_FFN);
    printf("============================================================\n\n");

    /* ── Load corpus ── */
    size_t corpus_len;
    unsigned char *corpus = load_corpus("data/corpus.txt", &corpus_len);
    printf("Corpus: %zu bytes  |  Unique chars: ", corpus_len);
    int seen[256] = {0};
    for (size_t i = 0; i < corpus_len; i++) seen[corpus[i]] = 1;
    int n_unique = 0;
    for (int i = 0; i < 256; i++) n_unique += seen[i];
    printf("%d\n\n", n_unique);

    /* ── Model & optimizer ── */
    LMModel    *model = model_create();
    LMOptimizers *opt = opts_create(model);
    LMGradients   *gr = grads_create(model);

    int H = LM_HIDDEN, F = LM_FFN, V = LM_VOCAB, C = LM_CTX;

    /* ── Memory arena for training activations ──────────────────────────
     * Each step needs: x(C×H), out(C×H), logits(C×V), grad_logits(C×V),
     *                  grad_out(C×H), grad_x(C×H), ff1(C×F),
     *                  grad_attn_out(C×H)
     * Total per step ≈ C*(6H + 2V + F) floats
     * We allocate 4× that for safety.
     * ────────────────────────────────────────────────────────────────── */
    size_t arena_sz = (size_t)C * (6*H + 2*V + F) * 4;
    TensorArena *arena = arena_create(arena_sz);

    printf("Memory Arena: %.2f MB pre-allocated for training activations\n",
           arena_sz * sizeof(float) / (1024.0 * 1024.0));
    printf("  (replaces ~%d malloc/free calls per step)\n\n",
           8 * LM_BATCH);

    /* Loss curve for CSV */
    FILE *loss_csv = fopen("results/metrics/lm_training_loss.csv", "w");
    if (loss_csv) fprintf(loss_csv, "epoch,avg_loss,perplexity\n");

    double total_train_ms = 0.0;

    for (int epoch = 0; epoch < LM_EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        int    n_steps    = 0;
        double epoch_ms   = 0.0;

        /* Shuffle starting positions */
        int max_start = (int)corpus_len - C - 1;

        for (int step = 0; step < LM_BATCH; step++) {
            /* Random context window from corpus */
            int start = rand() % max_start;

            int input_ids[LM_CTX];
            int target_id;
            for (int i = 0; i < C; i++)
                input_ids[i] = (int)corpus[start + i];
            target_id = (int)corpus[start + C];  /* next character */

            /* Full causal mask */
            int mask[LM_CTX];
            for (int i = 0; i < C; i++) mask[i] = 1;

            /* Build target: predict next token at the LAST position only */
            int target[LM_CTX];
            for (int i = 0; i < C - 1; i++) target[i] = -1;   /* ignore */
            target[C - 1] = target_id;                          /* predict this */

            /* ── ARENA ALLOCATION: all activations from pool ── */
            arena_reset(arena);
            Tensor *x          = arena_tensor(arena, C, H);
            Tensor *out        = arena_tensor(arena, C, H);
            Tensor *logits     = arena_tensor(arena, C, V);
            Tensor *grad_logits= arena_tensor(arena, C, V);
            Tensor *grad_out   = arena_tensor(arena, C, H);
            Tensor *grad_x     = arena_tensor(arena, C, H);

            double t0 = now_ms();

            /* ── Forward ── */
            embedding_forward(model->emb, input_ids, C, x);
            for (int i = 0; i < C; i++)
                x->data[i * H + (i % H)] += 0.01f;

            Tensor *ff1 = NULL;  /* ff1 is created inside transformer_block */
            AttentionCache cache = {0};

            transformer_block_forward(
                x, model->Wq, model->Wk, model->Wv, model->Wo,
                model->W1, model->b1, model->W2, model->b2,
                out, &ff1, &cache, mask
            );
            linear_forward(out, model->W_lm, model->b_lm, logits);

            float loss = cross_entropy_loss(logits, target);
            epoch_loss += loss;
            n_steps++;

            cross_entropy_loss_grad(logits, target, grad_logits);

            /* ── Backward ── */
            tensor_zero(gr->grad_Wlm); tensor_zero(gr->grad_blm);
            linear_backward(out, model->W_lm, grad_logits,
                            grad_out, gr->grad_Wlm, gr->grad_blm);

            tensor_zero(gr->grad_Wq); tensor_zero(gr->grad_Wk);
            tensor_zero(gr->grad_Wv); tensor_zero(gr->grad_Wo);
            tensor_zero(gr->grad_W1); tensor_zero(gr->grad_b1);
            tensor_zero(gr->grad_W2); tensor_zero(gr->grad_b2);

            Tensor *grad_attn_out = arena_tensor(arena, C, H);

            ffn_backward(
                cache.attn_out, ff1, grad_out,
                model->W2, model->W1,
                gr->grad_W2, gr->grad_b2,
                gr->grad_W1, gr->grad_b1,
                grad_attn_out
            );

            attention_backward(
                x, model->Wq, model->Wk, model->Wv, model->Wo,
                grad_attn_out, &cache,
                grad_x,
                gr->grad_Wq, gr->grad_Wk, gr->grad_Wv, gr->grad_Wo
            );

            tensor_zero(gr->grad_emb);
            embedding_backward(model->emb, input_ids, C, grad_x, gr->grad_emb);

            epoch_ms += now_ms() - t0;

            /* ── Gradient clipping ── */
            clip_grad(gr->grad_Wq,  LM_GRAD_CLIP);
            clip_grad(gr->grad_Wk,  LM_GRAD_CLIP);
            clip_grad(gr->grad_Wv,  LM_GRAD_CLIP);
            clip_grad(gr->grad_Wo,  LM_GRAD_CLIP);
            clip_grad(gr->grad_W1,  LM_GRAD_CLIP);
            clip_grad(gr->grad_b1,  LM_GRAD_CLIP);
            clip_grad(gr->grad_W2,  LM_GRAD_CLIP);
            clip_grad(gr->grad_b2,  LM_GRAD_CLIP);
            clip_grad(gr->grad_Wlm, LM_GRAD_CLIP);
            clip_grad(gr->grad_blm, LM_GRAD_CLIP);
            clip_grad(gr->grad_emb, LM_GRAD_CLIP);

            /* ── Adam updates ── */
            adam_step(model->Wq,          gr->grad_Wq,  opt->opt_Wq);
            adam_step(model->Wk,          gr->grad_Wk,  opt->opt_Wk);
            adam_step(model->Wv,          gr->grad_Wv,  opt->opt_Wv);
            adam_step(model->Wo,          gr->grad_Wo,  opt->opt_Wo);
            adam_step(model->W1,          gr->grad_W1,  opt->opt_W1);
            adam_step(model->b1,          gr->grad_b1,  opt->opt_b1);
            adam_step(model->W2,          gr->grad_W2,  opt->opt_W2);
            adam_step(model->b2,          gr->grad_b2,  opt->opt_b2);
            adam_step(model->W_lm,        gr->grad_Wlm, opt->opt_Wlm);
            adam_step(model->b_lm,        gr->grad_blm, opt->opt_blm);
            adam_step(model->emb->weights,gr->grad_emb, opt->opt_emb);

            /* ── Free non-arena memory (cache + ff1 from transformer_block) ── */
            tensor_free(cache.Q);     tensor_free(cache.K);
            tensor_free(cache.V);     tensor_free(cache.scores);
            tensor_free(cache.attn);  tensor_free(cache.attn_out);
            if (ff1) tensor_free(ff1);
        }

        double avg_loss = epoch_loss / n_steps;
        double perplexity = exp(avg_loss);
        total_train_ms += epoch_ms;

        printf("Epoch %3d/%d | Loss = %.4f | Perplexity = %6.2f | %.1f ms\n",
               epoch + 1, LM_EPOCHS, avg_loss, perplexity, epoch_ms);

        if (loss_csv)
            fprintf(loss_csv, "%d,%.4f,%.4f\n", epoch + 1, avg_loss, perplexity);
    }

    if (loss_csv) fclose(loss_csv);

    printf("\n--- Training Complete ---\n");
    printf("Total training time : %.2f s\n", total_train_ms / 1000.0);
    printf("Time per epoch      : %.1f ms\n", total_train_ms / LM_EPOCHS);

    /* ── Arena stats ── */
    printf("\nMemory Arena Stats:\n");
    arena_report(arena);

    /* ── Generate text ── */
    printf("\n============================================================\n");
    printf(" Text Generation (Temperature = 0.8)\n");
    printf("============================================================\n");
    generate(model, "the cat", LM_GEN_LEN);
    generate(model, "a bright", LM_GEN_LEN);

    /* ── Memory footprint ── */
    printf("\n============================================================\n");
    printf(" Model Memory Footprint\n");
    printf("============================================================\n");
    size_t fp32_params = (size_t)(LM_HIDDEN*LM_HIDDEN*4 +
                                  LM_HIDDEN*LM_FFN*2   +
                                  LM_HIDDEN*LM_VOCAB);
    size_t fp32_bytes = fp32_params * sizeof(float);
    size_t int8_bytes = fp32_params;
    printf("  Embedding table:     %.2f MB (FP32)\n",
           (LM_VOCAB * LM_HIDDEN * sizeof(float)) / (1024.0*1024.0));
    printf("  Transformer weights: %.2f MB (FP32)  ->  %.2f MB (INT8)  [%.1fx]\n",
           fp32_bytes / (1024.0*1024.0),
           int8_bytes / (1024.0*1024.0),
           (float)fp32_bytes / (float)int8_bytes);

    /* ── Cleanup ── */
    arena_free(arena);
    embedding_free(model->emb);
    tensor_free(model->Wq); tensor_free(model->Wk);
    tensor_free(model->Wv); tensor_free(model->Wo);
    tensor_free(model->W1); tensor_free(model->b1);
    tensor_free(model->W2); tensor_free(model->b2);
    tensor_free(model->W_lm); tensor_free(model->b_lm);
    free(model);
    free(corpus);

    printf("\nLoss curve saved to: results/metrics/lm_training_loss.csv\n");
    return 0;
}
