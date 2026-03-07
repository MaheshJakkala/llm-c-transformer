// #include "tensor.h"
// #include "loss.h"
// #include <math.h>
// #include <stdio.h>

// float cross_entropy_loss(const Tensor *logits, const int *target){
//     int rows = logits->rows;
//     int cols = logits->cols;
//     float loss = 0.0f;

//     for(int i=0;i<rows;i++){
//         if (target[i] == -1) continue;
//         int t = target[i];  // correct class index
//         float max_val = logits->data[i*cols];
//         for(int j=1;j<cols;j++)
//             if(logits->data[i*cols + j] > max_val) max_val = logits->data[i*cols + j];

//         float sum_exp = 0.0f;
//         for(int j=0;j<cols;j++)
//             sum_exp += expf(logits->data[i*cols + j] - max_val);

//         float log_prob = logits->data[i*cols + t] - max_val - logf(sum_exp);
//         loss -= log_prob;
//     }

//     return loss / rows;
// }

// void cross_entropy_loss_grad(const Tensor *logits, const int *target, Tensor *grad){
//     int rows = logits->rows, cols = logits->cols;
//     for(int i=0;i<rows;i++){
//         float sum_exp = 0.0f;
//         float max_val = logits->data[i*cols];
//         for(int j=1;j<cols;j++)
//             if(logits->data[i*cols + j] > max_val) max_val = logits->data[i*cols + j];

//         for(int j=0;j<cols;j++)
//             sum_exp += expf(logits->data[i*cols + j] - max_val);

//         for(int j=0;j<cols;j++){
//             float s = expf(logits->data[i*cols+j]-max_val)/sum_exp;
//             grad->data[i*cols+j] = s - (j==target[i]?1.0f:0.0f);
//             grad->data[i*cols+j] /= rows; // normalize
//         }
//     }
// }



#include "tensor.h"
#include "loss.h"
#include <math.h>
#include <stdio.h>

/*
  logits: [seq_len, num_classes]
  target: [seq_len]
  target[i] == -1  → ignore (PAD)
*/

float cross_entropy_loss(const Tensor *logits, const int *target) {
    int rows = logits->rows;
    int cols = logits->cols;

    float loss = 0.0f;
    int valid_count = 0;

    for (int i = 0; i < rows; i++) {
        if (target[i] == -1) continue;

        int t = target[i];

        if (t < 0 || t >= cols) {
    printf("BAD TARGET: %d (cols=%d)\n", t, cols);
    exit(1);
}


        /* numerical stability */
        float max_val = logits->data[i * cols];
        for (int j = 1; j < cols; j++) {
            float v = logits->data[i * cols + j];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(logits->data[i * cols + j] - max_val);
        }

        float log_prob =
            logits->data[i * cols + t] - max_val - logf(sum_exp);

        loss -= log_prob;
        valid_count++;
    }

    if (valid_count == 0) return 0.0f;
    return loss / valid_count;
}


void cross_entropy_loss_grad(
    const Tensor *logits,
    const int *target,
    Tensor *grad
) {
    int rows = logits->rows;
    int cols = logits->cols;

    tensor_zero(grad);

    int valid_count = 0;
    for (int i = 0; i < rows; i++)
        if (target[i] != -1)
            valid_count++;

    if (valid_count == 0) return;

    for (int i = 0; i < rows; i++) {
        if (target[i] == -1) continue;

        float max_val = logits->data[i * cols];
        for (int j = 1; j < cols; j++) {
            float v = logits->data[i * cols + j];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum_exp += expf(logits->data[i * cols + j] - max_val);
        }

        for (int j = 0; j < cols; j++) {
            float softmax =
                expf(logits->data[i * cols + j] - max_val) / sum_exp;

            grad->data[i * cols + j] =
                (softmax - (j == target[i])) / valid_count;
        }
    }
}
