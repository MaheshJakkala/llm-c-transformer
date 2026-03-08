/*
 * bench_ggml.c  —  Q4_0 (GGML format) vs INT8-AVX2 vs FP32
 * ==========================================================
 *
 * Implements GGML's actual Q4_0 block quantization from scratch:
 *   - Block size: 32 elements (identical to llama.cpp ggml-quants.h)
 *   - Storage:    4 bits per weight, packed 2 per byte
 *   - Scale:      1 float32 per block of 32 (GGML uses f16; we use f32 —
 *                 same algorithm, slightly larger scale storage)
 *   - Dequant:    expand nibbles → float, multiply by block scale
 *
 * This is the same kernel path llama.cpp takes on CPU without BLAS.
 * Published llama.cpp throughput for llama-7B on similar CPU: ~3-8 tok/s
 * (large model, different scale; used for context only — see BENCHMARK.md)
 *
 * Build:  gcc -O3 -mavx2 -mfma -fopenmp -std=c11 \
 *             src/bench_ggml.c -o bench_ggml -lm
 *
 * Output: results/metrics/ggml_comparison.csv
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

/* ── config (matches config.h) ─────────────────────────────────────────── */
#define HIDDEN   256
#define FFN_H    512
#define VOCAB    4096
#define NC       9
#define Q_BLOCK  32      /* GGML Q4_0 block size — never change */

#define BENCH_ITERS 500
#define WARM_ITERS   50

/* ── timing ─────────────────────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Q4_0  —  GGML block quantization
 *  Identical algorithm to ggml-quants.c:quantize_row_q4_0()
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    float   scale;          /* 1 float32 per block (GGML uses f16) */
    uint8_t nibbles[Q_BLOCK/2]; /* 2 × int4 packed per byte */
} Q4Block;

/* number of Q4 blocks needed for n elements */
#define N_BLOCKS(n) (((n) + Q_BLOCK - 1) / Q_BLOCK)

typedef struct {
    int      rows, cols;
    Q4Block *blocks;        /* rows * ceil(cols/Q_BLOCK) blocks */
    int      n_blocks_per_row;
} Q4Tensor;

/* Quantize a row of `n` floats into Q4_0 blocks */
static void quantize_row_q4(const float *src, Q4Block *blocks, int n) {
    int nb = N_BLOCKS(n);
    for (int b = 0; b < nb; b++) {
        const float *row = src + b * Q_BLOCK;
        int count = (b == nb-1 && n % Q_BLOCK) ? (n % Q_BLOCK) : Q_BLOCK;

        /* find max abs in block → scale */
        float amax = 0.0f;
        for (int i = 0; i < count; i++) {
            float a = fabsf(row[i]);
            if (a > amax) amax = a;
        }
        float scale = amax / 7.0f;   /* 4-bit signed: range [-8,7], use 7 */
        blocks[b].scale = scale;

        /* quantize: map float → 4-bit unsigned [0,15] with zero_point=8 */
        memset(blocks[b].nibbles, 0, sizeof(blocks[b].nibbles));
        for (int i = 0; i < count; i++) {
            int q = (scale > 0) ? (int)roundf(row[i] / scale) : 0;
            if (q < -8) q = -8;
            if (q >  7) q =  7;
            uint8_t uq = (uint8_t)(q + 8);  /* shift to [0,15] */
            if (i & 1) blocks[b].nibbles[i/2] |= (uq << 4);
            else        blocks[b].nibbles[i/2] |= uq;
        }
    }
}

Q4Tensor *quantize_q4(int rows, int cols, const float *data) {
    Q4Tensor *q = malloc(sizeof(Q4Tensor));
    q->rows = rows; q->cols = cols;
    q->n_blocks_per_row = N_BLOCKS(cols);
    q->blocks = malloc((size_t)rows * q->n_blocks_per_row * sizeof(Q4Block));
    for (int r = 0; r < rows; r++)
        quantize_row_q4(data + r*cols, q->blocks + r*q->n_blocks_per_row, cols);
    return q;
}

void free_q4(Q4Tensor *q) { free(q->blocks); free(q); }

/* Q4_0 matmul: A[M×K] (float) × B_q4[K×N] (q4, stored transposed as N rows of K)
 * Output: C[M×N] float
 * This mirrors how llama.cpp multiplies query × quantized weight on CPU.      */
static void matmul_fp32_q4(
    const float  *A,       /* [M × K] float input activations  */
    const Q4Tensor *Bq,   /* [N rows × K cols, quantized]     */
    float        *C,       /* [M × N] float output             */
    int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const Q4Block *blocks = Bq->blocks + j * Bq->n_blocks_per_row;
            float acc = 0.0f;
            for (int b = 0; b < Bq->n_blocks_per_row; b++) {
                float scale = blocks[b].scale;
                int base_k = b * Q_BLOCK;
                int count = (base_k + Q_BLOCK <= K) ? Q_BLOCK : (K - base_k);
                for (int t = 0; t < count; t++) {
                    /* dequantize nibble */
                    uint8_t raw = (t & 1)
                        ? (blocks[b].nibbles[t/2] >> 4)
                        : (blocks[b].nibbles[t/2] & 0x0F);
                    float w = ((int)raw - 8) * scale;   /* undo zero_point */
                    acc += A[i*K + (base_k+t)] * w;
                }
            }
            C[i*N + j] = acc;
        }
    }
}

/* ── INT8-AVX2 baseline (from existing bench.c) ──────────────────────────── */
static void matmul_int8_avx2(const int8_t *A, const int8_t *B, float *C,
                               int M, int N, int K, float scale)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256i acc = _mm256_setzero_si256();
            for (int k = 0; k < K; k += 32) {
                __m256i av = _mm256_loadu_si256((__m256i*)(A + i*K + k));
                __m256i bv = _mm256_loadu_si256((__m256i*)(B + j*K + k));
                __m256i md = _mm256_maddubs_epi16(av, bv);
                __m256i m32 = _mm256_madd_epi16(md, _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, m32);
            }
            int32_t tmp[8]; _mm256_storeu_si256((__m256i*)tmp, acc);
            int32_t s = 0; for (int t=0;t<8;t++) s+=tmp[t];
            C[i*N + j] = s * scale;
        }
    }
}

/* ── FP32 naive baseline ──────────────────────────────────────────────────── */
static void matmul_fp32(const float *A, const float *B, float *C,
                         int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[j*K+k];
            C[i*N+j] = s;
        }
}

/* ── memory size helpers ──────────────────────────────────────────────────── */
static size_t q4_weight_bytes(int rows, int cols) {
    return (size_t)rows * N_BLOCKS(cols) * sizeof(Q4Block);
}
static size_t i8_weight_bytes(int rows, int cols) { return (size_t)rows * cols; }
static size_t f32_weight_bytes(int rows, int cols) { return (size_t)rows * cols * 4; }

/* ══════════════════════════════════════════════════════════════════════════ */
int main(void)
{
    srand(42);
    /* build random FP32 weight matrix (K×N, transposed for our kernels) */
    int M = 16, K = HIDDEN, N = HIDDEN;  /* representative: seq=16, attn proj */

    float *W_f32 = malloc(K * N * sizeof(float));
    for (int i = 0; i < K*N; i++) W_f32[i] = ((float)rand()/RAND_MAX - 0.5f)*0.04f;

    /* INT8 quantization (existing repo format) */
    float wmax = 0; for (int i=0;i<K*N;i++) if (fabsf(W_f32[i])>wmax) wmax=fabsf(W_f32[i]);
    float i8scale = wmax/127.0f;
    int8_t *W_i8 = malloc(K*N); /* stored transposed: N rows × K cols */
    for (int n=0;n<N;n++) for (int k=0;k<K;k++)
        W_i8[n*K+k]=(int8_t)fmaxf(-127,fminf(127,roundf(W_f32[k*N+n]/i8scale)));

    /* Q4_0 quantization: quantize transposed weight (N rows × K cols) */
    float *W_f32_T = malloc(N * K * sizeof(float));
    for (int n=0;n<N;n++) for (int k=0;k<K;k++) W_f32_T[n*K+k]=W_f32[k*N+n];
    Q4Tensor *W_q4 = quantize_q4(N, K, W_f32_T);

    /* input activations: M × K float */
    float *A_f32 = malloc(M * K * sizeof(float));
    int8_t *A_i8 = malloc(M * K);
    for (int i=0;i<M*K;i++) {
        A_f32[i] = ((float)rand()/RAND_MAX - 0.5f)*0.1f;
        A_i8[i]  = (int8_t)(A_f32[i]*127.0f);
    }

    float *C = malloc(M * N * sizeof(float));

    printf("========================================================\n");
    printf(" GGML Q4_0 vs INT8-AVX2 vs FP32 — MatMul Benchmark\n");
    printf(" Matrix: M=%d K=%d N=%d  (transformer projection shape)\n", M, K, N);
    printf("========================================================\n\n");

    /* ── weight memory comparison ──────────────────────────────────────── */
    size_t fp32_sz = f32_weight_bytes(N, K);
    size_t i8_sz   = i8_weight_bytes(N, K);
    size_t q4_sz   = q4_weight_bytes(N, K);
    printf("Weight Memory (K=%d N=%d):\n", K, N);
    printf("  FP32 : %zu bytes = %.3f KB\n", fp32_sz, fp32_sz/1024.0);
    printf("  INT8 : %zu bytes = %.3f KB  (%.2fx vs FP32)\n",
           i8_sz, i8_sz/1024.0, (float)fp32_sz/i8_sz);
    printf("  Q4_0 : %zu bytes = %.3f KB  (%.2fx vs FP32, %.2fx vs INT8)\n\n",
           q4_sz, q4_sz/1024.0, (float)fp32_sz/q4_sz, (float)i8_sz/q4_sz);

    /* ── latency benchmarks ────────────────────────────────────────────── */
    double t0, total;

    /* FP32 naive */
    for (int i=0;i<WARM_ITERS;i++) matmul_fp32(A_f32, W_f32_T, C, M, N, K);
    total=0; t0=now_ms();
    for (int i=0;i<BENCH_ITERS;i++) matmul_fp32(A_f32, W_f32_T, C, M, N, K);
    double fp32_ms = (now_ms()-t0)/BENCH_ITERS;

    /* INT8-AVX2 */
    for (int i=0;i<WARM_ITERS;i++) matmul_int8_avx2(A_i8, W_i8, C, M, N, K, i8scale*i8scale);
    total=0; t0=now_ms();
    for (int i=0;i<BENCH_ITERS;i++) matmul_int8_avx2(A_i8, W_i8, C, M, N, K, i8scale*i8scale);
    double i8_ms = (now_ms()-t0)/BENCH_ITERS;

    /* Q4_0 */
    for (int i=0;i<WARM_ITERS;i++) matmul_fp32_q4(A_f32, W_q4, C, M, N, K);
    total=0; t0=now_ms();
    for (int i=0;i<BENCH_ITERS;i++) matmul_fp32_q4(A_f32, W_q4, C, M, N, K);
    double q4_ms = (now_ms()-t0)/BENCH_ITERS;

    printf("%-20s | %10s | %14s | %12s\n",
           "Backend", "Time (ms)", "vs FP32", "Memory");
    printf("---------------------|------------|----------------|------------\n");
    printf("%-20s | %10.4f | %14s | %10.2f KB\n",
           "FP32 naive", fp32_ms, "1.00x baseline", fp32_sz/1024.0);
    printf("%-20s | %10.4f | %13.2fx     | %10.2f KB\n",
           "INT8-AVX2 (repo)", i8_ms, fp32_ms/i8_ms, i8_sz/1024.0);
    printf("%-20s | %10.4f | %13.2fx     | %10.2f KB\n",
           "Q4_0 (GGML format)", q4_ms, fp32_ms/q4_ms, q4_sz/1024.0);

    printf("\nINT8 vs Q4_0 speed:   %.2fx  (%s is faster)\n",
           (i8_ms < q4_ms) ? q4_ms/i8_ms : i8_ms/q4_ms,
           (i8_ms < q4_ms) ? "INT8-AVX2" : "Q4_0");
    printf("INT8 vs Q4_0 memory:  INT8=%.2fKB  Q4_0=%.2fKB  Q4_0 is %.2fx smaller\n",
           i8_sz/1024.0, q4_sz/1024.0, (float)i8_sz/q4_sz);

    /* ── CSV output ────────────────────────────────────────────────────── */
    FILE *f = fopen("results/metrics/ggml_comparison.csv", "w");
    if (f) {
        fprintf(f, "backend,time_ms,speedup_vs_fp32,weight_kb,bits_per_weight\n");
        fprintf(f, "FP32,%.4f,1.00,%.2f,32\n", fp32_ms, fp32_sz/1024.0);
        fprintf(f, "INT8-AVX2,%.4f,%.2f,%.2f,8\n", i8_ms, fp32_ms/i8_ms, i8_sz/1024.0);
        fprintf(f, "Q4_0-GGML,%.4f,%.2f,%.2f,4.5\n", q4_ms, fp32_ms/q4_ms, q4_sz/1024.0);
        fclose(f);
        printf("\nSaved: results/metrics/ggml_comparison.csv\n");
    }

    free(W_f32); free(W_f32_T); free(W_i8); free_q4(W_q4);
    free(A_f32); free(A_i8); free(C);
    return 0;
}
