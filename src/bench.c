/*
 * bench.c — Benchmark: FP32 vs INT8-AVX2 inference
 *
 * Measures:
 *  1. FP32  forward-pass latency (ms) and throughput (tokens/s)
 *  2. INT8-AVX2 forward-pass latency and throughput
 *  3. Memory footprint: FP32 weights vs INT8 weights
 *  4. Speedup ratio INT8 / FP32
 *
 * Outputs machine-readable CSV for plotting.
 */
#include "main.h"
#include <string.h>

#define BENCH_ITERS   50      /* warm-up + timed iterations */
#define WARM_ITERS     5

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ─── FP32 single-sample forward pass ─────────────────────────────────── */
static double bench_fp32(
    Embedding *emb,
    Tensor *Wq, Tensor *Wk, Tensor *Wv, Tensor *Wo,
    Tensor *W1, Tensor *b1, Tensor *W2, Tensor *b2,
    Tensor *W_cls, Tensor *b_cls,
    int *input_ids, int seq_len, int *padding_mask
) {
    double total = 0.0;

    for (int iter = 0; iter < BENCH_ITERS + WARM_ITERS; iter++) {
        Tensor *x      = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *out    = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *logits = tensor_create(seq_len, NUM_CLASSES);
        Tensor *ff1    = NULL;
        AttentionCache cache = {0};

        embedding_forward(emb, input_ids, seq_len, x);
        for (int i = 0; i < seq_len; i++)
            x->data[i * HIDDEN_SIZE + (i % HIDDEN_SIZE)] += 0.01f;

        double t0 = now_ms();

        transformer_block_forward(
            x, Wq, Wk, Wv, Wo,
            W1, b1, W2, b2,
            out, &ff1, &cache, padding_mask
        );
        linear_forward(out, W_cls, b_cls, logits);

        double t1 = now_ms();

        if (iter >= WARM_ITERS) total += (t1 - t0);

        tensor_free(cache.Q); tensor_free(cache.K); tensor_free(cache.V);
        tensor_free(cache.scores); tensor_free(cache.attn); tensor_free(cache.attn_out);
        tensor_free(x); tensor_free(out); tensor_free(logits);
        if (ff1) tensor_free(ff1);
    }

    return total / BENCH_ITERS;
}

/* ─── INT8-AVX2 single-sample forward pass ─────────────────────────────── */
static double bench_int8(
    Embedding *emb,
    QTensor *Wq_q, QTensor *Wk_q, QTensor *Wv_q, QTensor *Wo_q,
    QTensor *Wcls_q, Tensor *b_cls,
    int *input_ids, int seq_len, int *padding_mask
) {
    double total = 0.0;

    for (int iter = 0; iter < BENCH_ITERS + WARM_ITERS; iter++) {
        Tensor *x   = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *out = tensor_create(seq_len, HIDDEN_SIZE);
        Tensor *logits = tensor_create(seq_len, NUM_CLASSES);
        AttentionCache cache = {0};

        embedding_forward(emb, input_ids, seq_len, x);

        double t0 = now_ms();

        QTensor *x_q = quantize_tensor(x);

        attention_forward_q8(
            x_q, Wq_q, Wk_q, Wv_q, Wo_q,
            padding_mask, out, &cache
        );

        QTensor *out_q = quantize_tensor(out);
        linear_forward_q8(out_q, Wcls_q, b_cls, logits);
        free_qtensor(out_q);
        free_qtensor(x_q);

        double t1 = now_ms();

        if (iter >= WARM_ITERS) total += (t1 - t0);

        tensor_free(cache.Q); tensor_free(cache.K); tensor_free(cache.V);
        tensor_free(cache.scores); tensor_free(cache.attn); tensor_free(cache.attn_out);
        tensor_free(x); tensor_free(out); tensor_free(logits);
    }

    return total / BENCH_ITERS;
}

int main(void) {
    srand(42);

    printf("============================================================\n");
    printf(" LLM-C Transformer — Inference Benchmark\n");
    printf(" FP32 vs INT8-AVX2\n");
    printf("============================================================\n\n");

    int hidden = HIDDEN_SIZE;
    int ffn_h  = FFN_HIDDEN;
    int nc     = NUM_CLASSES;

    /* ── Allocate and fill weights ── */
    Tokenizer *tok = tokenizer_create();
    Embedding *emb = embedding_create(tok->vocab_size, hidden);

    Tensor *Wq = tensor_create(hidden, hidden); tensor_fill_random(Wq, -0.02f, 0.02f);
    Tensor *Wk = tensor_create(hidden, hidden); tensor_fill_random(Wk, -0.02f, 0.02f);
    Tensor *Wv = tensor_create(hidden, hidden); tensor_fill_random(Wv, -0.02f, 0.02f);
    Tensor *Wo = tensor_create(hidden, hidden); tensor_fill_random(Wo, -0.02f, 0.02f);
    Tensor *W1 = tensor_create(hidden, ffn_h);  tensor_fill_random(W1, -0.02f, 0.02f);
    Tensor *b1 = tensor_create(1,      ffn_h);  tensor_zero(b1);
    Tensor *W2 = tensor_create(ffn_h,  hidden); tensor_fill_random(W2, -0.02f, 0.02f);
    Tensor *b2 = tensor_create(1,      hidden); tensor_zero(b2);
    Tensor *Wcls = tensor_create(hidden, nc);    tensor_fill_random(Wcls, -0.02f, 0.02f);
    Tensor *bcls = tensor_create(1, nc);         tensor_zero(bcls);

    /* ── INT8-quantized weights (quantize once, reuse) ── */
    QTensor *Wq_q  = quantize_weight_transpose(Wq);
    QTensor *Wk_q  = quantize_weight_transpose(Wk);
    QTensor *Wv_q  = quantize_weight_transpose(Wv);
    QTensor *Wo_q  = quantize_weight_transpose(Wo);
    QTensor *Wcls_q = quantize_weight_transpose(Wcls);

    /* ── Memory footprint ── */
    size_t fp32_params = (size_t)(hidden * hidden * 4 +   /* Wq,Wk,Wv,Wo */
                                  hidden * ffn_h * 2 +    /* W1,W2 */
                                  hidden * nc);           /* Wcls */
    size_t fp32_bytes  = fp32_params * sizeof(float);
    size_t int8_bytes  = fp32_params * sizeof(int8_t);
    float  mem_ratio   = (float)fp32_bytes / (float)int8_bytes;

    printf("Weight Memory (FP32): %.2f MB\n", fp32_bytes  / (1024.0 * 1024.0));
    printf("Weight Memory (INT8): %.2f MB\n", int8_bytes  / (1024.0 * 1024.0));
    printf("Memory Reduction:     %.2fx\n\n", mem_ratio);

    /* ── Benchmark across different sequence lengths ── */
    int seq_lens[] = {8, 16, 32, 64};
    int n_lens = 4;

    printf("%-8s | %-12s | %-12s | %-10s | %-14s | %-14s\n",
           "SeqLen", "FP32 (ms)", "INT8 (ms)", "Speedup",
           "FP32 tok/s", "INT8 tok/s");
    printf("---------|--------------|--------------|------------|----------------|----------------\n");

    /* CSV output */
    FILE *csv = fopen("results/metrics/bench_results.csv", "w");
    if (csv) {
        fprintf(csv, "seq_len,fp32_ms,int8_ms,speedup,fp32_toks,int8_toks,"
                     "fp32_mem_mb,int8_mem_mb,mem_reduction\n");
    }

    for (int li = 0; li < n_lens; li++) {
        int seq = seq_lens[li];

        /* Fake input */
        int input_ids[MAX_SEQ_LEN];
        for (int i = 0; i < seq; i++) input_ids[i] = (i * 37 + 13) % VOCAB_SIZE;
        int padding_mask[MAX_SEQ_LEN];
        for (int i = 0; i < seq; i++) padding_mask[i] = 1;

        double fp32_ms = bench_fp32(
            emb, Wq, Wk, Wv, Wo, W1, b1, W2, b2, Wcls, bcls,
            input_ids, seq, padding_mask
        );
        double int8_ms = bench_int8(
            emb, Wq_q, Wk_q, Wv_q, Wo_q, Wcls_q, bcls,
            input_ids, seq, padding_mask
        );

        double speedup     = fp32_ms / int8_ms;
        double fp32_toks   = seq / (fp32_ms / 1000.0);
        double int8_toks   = seq / (int8_ms / 1000.0);

        printf("%-8d | %-12.3f | %-12.3f | %-10.2f | %-14.0f | %-14.0f\n",
               seq, fp32_ms, int8_ms, speedup, fp32_toks, int8_toks);

        if (csv) {
            fprintf(csv, "%d,%.4f,%.4f,%.4f,%.0f,%.0f,%.4f,%.4f,%.2f\n",
                    seq, fp32_ms, int8_ms, speedup, fp32_toks, int8_toks,
                    fp32_bytes / (1024.0 * 1024.0),
                    int8_bytes / (1024.0 * 1024.0),
                    mem_ratio);
        }
    }

    if (csv) fclose(csv);

    printf("\nResults saved to results/metrics/bench_results.csv\n");

    /* ── Matmul micro-benchmark ── */
    printf("\n--- INT8-AVX2 vs FP32 Matmul Micro-Benchmark (M=N=K=%d) ---\n", HIDDEN_SIZE);
    {
        int M = HIDDEN_SIZE, N = HIDDEN_SIZE, K = HIDDEN_SIZE;
        int8_t  *A_i8 = malloc(M * K);
        int8_t  *B_i8 = malloc(K * N);
        int32_t *C_i32 = malloc(M * N * sizeof(int32_t));
        float   *A_f32 = malloc(M * K * sizeof(float));
        float   *B_f32 = malloc(K * N * sizeof(float));
        float   *C_f32 = malloc(M * N * sizeof(float));

        for (int i = 0; i < M * K; i++) { A_i8[i] = (int8_t)(i % 127); A_f32[i] = A_i8[i] * 0.01f; }
        for (int i = 0; i < K * N; i++) { B_i8[i] = (int8_t)(i % 127); B_f32[i] = B_i8[i] * 0.01f; }

        int REPS = 500;
        /* INT8-AVX2 */
        double t0 = now_ms();
        for (int r = 0; r < REPS; r++)
            matmul_q8_avx2(A_i8, B_i8, C_i32, M, N, K);
        double int8_mat_ms = (now_ms() - t0) / REPS;

        /* FP32 naive */
        t0 = now_ms();
        for (int r = 0; r < REPS; r++) {
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    float s = 0;
                    for (int k = 0; k < K; k++)
                        s += A_f32[i*K+k] * B_f32[k*N+j];
                    C_f32[i*N+j] = s;
                }
        }
        double fp32_mat_ms = (now_ms() - t0) / REPS;

        printf("FP32 matmul:      %.4f ms  (%.1f MFLOP/s)\n",
               fp32_mat_ms, 2.0 * M * N * K / (fp32_mat_ms * 1e-3) / 1e6);
        printf("INT8-AVX2 matmul: %.4f ms  (%.1f MOPS/s)\n",
               int8_mat_ms, 2.0 * M * N * K / (int8_mat_ms * 1e-3) / 1e6);
        printf("Matmul speedup:   %.2fx\n", fp32_mat_ms / int8_mat_ms);

        /* Save matmul CSV */
        FILE *mcsv = fopen("results/metrics/matmul_bench.csv", "w");
        if (mcsv) {
            fprintf(mcsv, "backend,time_ms,mflops,speedup\n");
            fprintf(mcsv, "FP32,%.4f,%.1f,1.00\n",
                    fp32_mat_ms, 2.0 * M * N * K / (fp32_mat_ms * 1e-3) / 1e6);
            fprintf(mcsv, "INT8-AVX2,%.4f,%.1f,%.2f\n",
                    int8_mat_ms, 2.0 * M * N * K / (int8_mat_ms * 1e-3) / 1e6,
                    fp32_mat_ms / int8_mat_ms);
            fclose(mcsv);
        }

        free(A_i8); free(B_i8); free(C_i32);
        free(A_f32); free(B_f32); free(C_f32);
    }

    /* Cleanup */
    free_qtensor(Wq_q); free_qtensor(Wk_q); free_qtensor(Wv_q);
    free_qtensor(Wo_q); free_qtensor(Wcls_q);
    tensor_free(Wq); tensor_free(Wk); tensor_free(Wv); tensor_free(Wo);
    tensor_free(W1); tensor_free(b1); tensor_free(W2); tensor_free(b2);
    tensor_free(Wcls); tensor_free(bcls);
    embedding_free(emb);
    tokenizer_free(tok);

    return 0;
}
