#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include<math.h>
#include "qtensor.h"
#include "config.h"

typedef struct Tensor {
    int rows;
    int cols;
    float *data;
} Tensor;

Tensor *tensor_create(int rows, int cols);
void tensor_free(Tensor *t);
void tensor_fill_random(Tensor *t, float min_val, float max_val);
void tensor_zero(Tensor *t);

void tensor_add(const Tensor *x, const Tensor *y, Tensor *out);
void tensor_copy(const Tensor *src, Tensor *dst);
void tensor_print(Tensor *t);

void matmul_q8_q8_int32(
    const QTensor *A,
    const QTensor *B,
    int32_t *C
);

void matmul_q8_avx2(
    const int8_t *A,
    const int8_t *B,
    int32_t *C,
    int M, int N, int K
);


#endif
