#ifndef QTENSOR_H
#define QTENSOR_H

#include <stdint.h>
#include<math.h>
// #include "tensor.h"
typedef struct Tensor Tensor;

// typedef struct {
//     int8_t *data;
//     int rows, cols;
//     float scale;
// } QTensor;

typedef struct {
    int rows;
    int cols;
    int8_t *data;
    float scale;
} QTensor;


QTensor* quantize_tensor(const Tensor *src);
QTensor* quantize_weight_transpose(const Tensor *W);
void free_qtensor(QTensor *qt);

void matmul_q8(
    const QTensor *A,
    const QTensor *B,
    Tensor *out
);

#endif
