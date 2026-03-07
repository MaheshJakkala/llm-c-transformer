// #pragma omp parallel for collapse(2)
// tensor.c
#include "tensor.h"
#include "qtensor.h"


Tensor *tensor_create(int rows, int cols){
    if (rows <= 0 || cols <= 0) return NULL;
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) { fprintf(stderr, "tensor_create: malloc failed for Tensor struct\n"); return NULL; }
    t->rows = rows;
    t->cols = cols;
    t->data = (float*)malloc(sizeof(float) * rows * cols);
    // t->data = calloc(rows * cols, sizeof(float));

    if (!t->data) {
        fprintf(stderr, "tensor_create: malloc failed for Tensor data\n");
        free(t);
        return NULL;
    }
    // Initialize to zero
    for (int i = 0; i < rows * cols; ++i) t->data[i] = 0.0f;
    return t;
}

void tensor_zero(Tensor *t) {
    if (!t || !t->data){
        printf("Invalid tensor zero\n");
        return;
    }
    int n = t->rows * t->cols;
    for (int i = 0; i < n; ++i)
        t->data[i] = 0.0f;
}

void tensor_add(const Tensor *x, const Tensor *y, Tensor *out) {
    if (!x || !y || !out) {
        fprintf(stderr, "tensor_add: NULL tensor passed\n");
        return;
    }
    if (x->rows != y->rows || x->cols != y->cols ||
        out->rows != x->rows || out->cols != x->cols) {
        fprintf(stderr, "tensor_add: dimension mismatch\n");
        return;
    }

    int total = x->rows * x->cols;
    for (int i = 0; i < total; i++) {
        out->data[i] = x->data[i] + y->data[i];
    }
}
void tensor_copy(const Tensor *src, Tensor *dst) {
    if (!src || !dst) {
        fprintf(stderr, "tensor_copy: NULL tensor passed\n");
        return;
    }

    if (src->rows != dst->rows || src->cols != dst->cols) {
        fprintf(stderr, "tensor_copy: dimension mismatch (%dx%d vs %dx%d)\n",
                src->rows, src->cols, dst->rows, dst->cols);
        return;
    }

    int total = src->rows * src->cols;
    for (int i = 0; i < total; i++) {
        dst->data[i] = src->data[i];
    }
}

void tensor_fill_random(Tensor *t, float min_val, float max_val) {
    if (!t || !t->data) return;
    for (int i = 0; i < t->rows * t->cols; ++i) {
        float r = (float)rand() / (float)RAND_MAX; // 0..1
        t->data[i] = min_val + r * (max_val - min_val);
    }
}

void tensor_free(Tensor *t) {
    if (!t) return;
    if (t->data) free(t->data);
    free(t);
}

void tensor_print(Tensor *t){
    printf("Shape: %d x%d \n",t->rows,t->cols);
    for(int r=0;r<t->rows;r++){
        for(int c=0;c<t->cols;c++){
            printf("%.4f ", t->data[r * t->cols + c]);
        }
        printf("\n");
    }
}

#include <immintrin.h>

static inline int32_t horizontal_add(__m256i v) {
    __m128i vlow  = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    vlow = _mm_add_epi32(vlow, vhigh);

    vlow = _mm_hadd_epi32(vlow, vlow);
    vlow = _mm_hadd_epi32(vlow, vlow);

    return _mm_cvtsi128_si32(vlow);
}

void matmul_q8_q8_int32(
    const QTensor *A,
    const QTensor *B,
    int32_t *C
) {
    int M = A->rows;
    int K = A->cols;
    int N = B->rows;  // B is K x N but stored row-major

    // int block = 64;

// for (int ii = 0; ii < M; ii += block)
// for (int jj = 0; jj < N; jj += block)
// for (int kk = 0; kk < K; kk += block)

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int32_t sum = 0;

            // for (int k = 0; k < K; ++k) {
            //     int8_t a = A->data[i*K + k];
            //     int8_t b = B->data[j*K + k]; // B transposed logic
            //     sum += (int32_t)a * (int32_t)b;
            // }

            // for (int k = 0; k < K; k += 32) {

            //     __m256i a_vec = _mm256_loadu_si256((__m256i*)&A->data[i*K + k]);
            //     __m256i b_vec = _mm256_loadu_si256((__m256i*)&B->data[j*K + k]);

            //     __m256i madd = _mm256_maddubs_epi16(a_vec, b_vec);
            //     __m256i madd32 = _mm256_madd_epi16(madd, _mm256_set1_epi16(1));

            //     sum += horizontal_add(madd32);
            // }

            for (int k = 0; k < K; ++k) {
    sum += (int32_t)A->data[i*K + k] *
           (int32_t)B->data[j*K + k];
}


            C[i*N + j] = sum;
        }
    }
}




void matmul_q8_avx2(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int M, int N, int K)
{
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {

            __m256i acc = _mm256_setzero_si256();

            for (int k = 0; k < K; k += 32) {

                __m256i a_vec = _mm256_loadu_si256(
                    (__m256i const*)(A + i*K + k));

                __m256i b_vec = _mm256_loadu_si256(
                    (__m256i const*)(B + j*K + k));

                __m256i madd = _mm256_maddubs_epi16(a_vec, b_vec);
                __m256i madd32 = _mm256_madd_epi16(
                    madd, _mm256_set1_epi16(1));

                acc = _mm256_add_epi32(acc, madd32);
            }

            int32_t tmp[8];
            _mm256_storeu_si256((__m256i*)tmp, acc);

            int32_t sum = 0;
            for (int t = 0; t < 8; t++)
                sum += tmp[t];

            C[i*N + j] = sum;
        }
    }
}
