// ops.c
#include "ops.h"
#include <stdio.h>

void matmul_naive(const Tensor *A, const Tensor *B, Tensor *C) {
     if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "matmul dimension mismatch\n");
        exit(1);
    }
    size_t M = A->rows, K = A->cols, N = B->cols;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A->data[i*K + k] * B->data[k*N + j];
            }
            C->data[i*N + j] = sum;
        }
    }
}

// C = A x B^T where B is (N x K); B^T is (K x N)
void matmul_transposeB(const Tensor *A, const Tensor *B, Tensor *C) {
    if (A->cols != B->cols || A->rows != C->rows || B->rows != C->cols) {
        fprintf(stderr, "matmul_transposeB shape mismatch\n");
        return;
    }
    size_t M = A->rows, K = A->cols, N = B->rows;
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // B[j,k] stored as B->data[j*K + k]
                sum += A->data[i*K + k] * B->data[j*K + k];
            }
            C->data[i*N + j] = sum;
        }
    }
}