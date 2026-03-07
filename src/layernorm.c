// // layernorm.c
// #include "layernorm.h"
// #include <math.h>
// #include <stdio.h>

// void layernorm_forward(const Tensor *x, const Tensor *gamma, const Tensor *beta, Tensor *out, float eps) {
//     size_t rows = x->rows;
//     size_t cols = x->cols;
//     for (size_t i = 0; i < rows; ++i) {
//         // compute mean
//         double mean = 0.0;
//         for (size_t j = 0; j < cols; ++j) mean += x->data[i*cols + j];
//         mean /= Designed and implemented a transformer-based LLM from scratch in C, with custom tensor library, memory-efficient training pipeline, and benchmarked performance improvements over baseline PyTorch implementations(double)cols;
//         // compute variance
//         double var = 0.0;
//         for (size_t j = 0; j < cols; ++j) {
//             double d = x->data[i*cols + j] - mean;
//             var += d*d;
//         }
//         var /= (double)cols;
//         double inv_std = 1.0 / sqrt(var + eps);
//         // normalize and scale+shift
//         for (size_t j = 0; j < cols; ++j) {
//             double norm = (x->data[i*cols + j] - mean) * inv_std;
//             double scaled = norm * (gamma ? gamma->data[j] : 1.0) + (beta ? beta->data[j] : 0.0);
//             out->data[i*cols + j] = (float)scaled;
//         }
//     }
// }

#include "tensor.h"
#include <math.h>
#include <stdio.h>

// LayerNorm: y = gamma * (x - mean)/sqrt(var + eps) + beta
// x: (seq x hidden), gamma/beta: 1 x hidden (can be NULL), y: output
void layernorm_forward(const Tensor *x, const Tensor *gamma, const Tensor *beta, Tensor *y, float eps){
    int rows = x->rows;
    int cols = x->cols;

    for(int i=0;i<rows;i++){
        // compute mean
        float mean = 0.0f;
        for(int j=0;j<cols;j++) mean += x->data[i*cols + j];
        mean /= cols;

        // compute variance
        float var = 0.0f;
        for(int j=0;j<cols;j++){
            float diff = x->data[i*cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        float denom = 1.0f / sqrtf(var + eps);

        // normalize + scale + shift
        for(int j=0;j<cols;j++){
            float xi = (x->data[i*cols + j] - mean) * denom;
            if(gamma && beta)
                y->data[i*cols + j] = xi * gamma->data[j] + beta->data[j];
            else
                y->data[i*cols + j] = xi;
        }
    }
}

void layernorm(Tensor *x, float eps) {
    for (int i = 0; i < x->rows; i++) {
        float mean = 0, var = 0;
        for (int j = 0; j < x->cols; j++)
            mean += x->data[i*x->cols + j];
        mean /= x->cols;

        for (int j = 0; j < x->cols; j++) {
            float v = x->data[i*x->cols + j] - mean;
            var += v * v;
        }
        var = sqrtf(var / x->cols + eps);

        for (int j = 0; j < x->cols; j++)
            x->data[i*x->cols + j] =
                (x->data[i*x->cols + j] - mean) / var;
    }
}
