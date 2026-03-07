// src/linear.c
#include "linear.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>

// y = x * W + b
// x: (batch x in_dim)
// W: (in_dim x out_dim)
// b: (1 x out_dim) OR (out_dim x 1)  OR NULL
// y: (batch x out_dim) -- must be allocated
void linear_forward(const Tensor *x, const Tensor *W, const Tensor *b, Tensor *y) {
    // printf("linear_forward called\n");
    if (!x || !W || !y) {
        printf("linear_forward: NULL pointer (x=%p W=%p y=%p)\n", (void*)x, (void*)W, (void*)y);
        fprintf(stderr, "linear_forward: NULL pointer (x=%p W=%p y=%p)\n", (void*)x, (void*)W, (void*)y);
        exit(1);
    }
    // printf("linear_forward: x(%d,%d) W(%d,%d) y(%d,%d) b=%p\n",
        //    x->rows, x->cols, W->rows, W->cols, y->rows, y->cols, (void*)b);
    int batch = x->rows;
    int in_dim = x->cols;
    int w_rows = W->rows;
    int out_dim = W->cols;
    
    // quick shape/consistency checks
    if (in_dim != w_rows) {
        
        fprintf(stderr, "linear_forward: shape mismatch: x.cols=%d but W.rows=%d\n", in_dim, w_rows);
        exit(1);
    }

    // printf("x shape: (%d,%d) , wq shape: (%d,%d)\n",x->rows,x->cols,W->rows,W->cols);
    // printf("\nx shape: (%d,%d) , W_cls shape: (%d,%d) , y shape: (%d,%d)\n",batch,in_dim,w_rows,out_dim,y->rows,y->cols);
    // printf("linear_forward called 2\n");
    if (y->rows != batch || y->cols != out_dim) {
        fprintf(stderr, "linear_forward: output shape mismatch: expected y (%d x %d) got (%d x %d)\n",batch, out_dim, y->rows, y->cols);
        exit(1);
    }
    
    if (!x || !x->data || !W || !W->data || !y || !y->data) {
    fprintf(stderr, "linear_forward: NULL tensor pointer\n");
    return;
}
    if (in_dim <= 0 || out_dim <= 0 || batch <= 0) {
        fprintf(stderr, "linear_forward: invalid tensor shape (batch=%d, in_dim=%d, out_dim=%d)\n",
                batch, in_dim, out_dim);
        return;
    }
    // bias shape check (if present)
    // printf("linear_forward: x(%d,%d) W(%d,%d) y(%d,%d) b=%p\n",
        //    x->rows, x->cols, W->rows, W->cols, y->rows, y->cols, (void*)b);
    int bias_is_valid = 0;
    if (b) {
        if ((b->rows == 1 && b->cols == out_dim) || (b->rows == out_dim && b->cols == 1)) {
            bias_is_valid = 1;
        } else {
            fprintf(stderr, "linear_forward: bias shape mismatch: expected (1 x %d) or (%d x 1), got (%d x %d)\n",
                    out_dim, out_dim, b->rows, b->cols);
            exit(1);
        }
    }

    // Compute y = x * W + b
    // printf("linear_forward: batch=%d, in_dim=%d, out_dim=%d, bias=%s\n",
        //    batch, in_dim, out_dim, bias_is_valid ? "yes" : "no");
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < out_dim; ++j) {
            double sum = 0.0;
            // dot product
            for (int k = 0; k < in_dim; ++k) {
                sum += (double)x->data[i * in_dim + k] * (double)W->data[k * out_dim + j];
            }
            // bias handling: both (1 x out_dim) or (out_dim x 1) are supported
            float bias_val = 0.0f;
            if (b) {
                // if b is (1 x out_dim): index j
                // if b is (out_dim x 1): index j (since b->cols==1 and stored row-major)
                bias_val = b->data[j];
            }
            y->data[i * out_dim + j] = (float)(sum + bias_val);
            // printf("y[%d,%d] = %.6f (bias=%.6f)\n", i, j, y->data[i * out_dim + j], bias_val);
        }
    }
}
void linear_backward(
    const Tensor *x,          // (B x D_in)
    const Tensor *W,          // (D_in x D_out)
    const Tensor *grad_out,   // (B x D_out)
   Tensor *grad_x,           // (B x D_in) or NULL
    Tensor *grad_W,           // (D_in x D_out)
    Tensor *grad_b            // (1 x D_out) or NULL
){
    int B     = x->rows;
    int D_in  = x->cols;
    int D_out = grad_out->cols;

    /* ================= grad_W =================
       grad_W += x^T · grad_out
    */
    if (grad_W) {
        for (int i = 0; i < D_in; i++) {
            for (int j = 0; j < D_out; j++) {
                float sum = 0.0f;
                for (int b = 0; b < B; b++) {
                    sum += x->data[b*D_in + i] *
                           grad_out->data[b*D_out + j];
                }
                grad_W->data[i*D_out + j] += sum;
            }
        }
    }

    /* ================= grad_b =================
       grad_b += sum over batch
       (ONLY if grad_b != NULL)
    */
    if (grad_b) {
        for (int j = 0; j < D_out; j++) {
            float sum = 0.0f;
            for (int b = 0; b < B; b++) {
                sum += grad_out->data[b*D_out + j];
            }
            grad_b->data[j] += sum;
        }
    }

    /* ================= grad_x =================
       grad_x += grad_out · W^T
       (ONLY if grad_x != NULL)
    */
    if (grad_x) {
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < D_in; i++) {
                float sum = 0.0f;
                for (int j = 0; j < D_out; j++) {
                    sum += grad_out->data[b*D_out + j] *
                           W->data[i*D_out + j];
                }
                grad_x->data[b*D_in + i] += sum;
                // if (grad_x->data[i] > 1.0f) grad_x->data[i] = 1.0f;
                // if (grad_x->data[i] < -1.0f) grad_x->data[i] = -1.0f;

            }
        }
    }
}

// void linear_forward_q8(
//     const QTensor *x,
//     const QTensor *W,
//     Tensor *out
// ){
//     if (out->rows != x->rows || out->cols != W->cols) {
//         fprintf(stderr, "linear_forward_q8: shape mismatch\n");
//         exit(1);
//     }

//     for (int i = 0; i < x->rows; i++) {
//         for (int j = 0; j < W->cols; j++) {
//             int32_t acc = 0;
//             for (int k = 0; k < x->cols; k++) {
//                 acc += x->data[i*x->cols+k] *
//                        W->data[k*W->cols+j];
//             }
//             out->data[i*out->cols+j] =
//                 acc * x->scale * W->scale;
//         }
//     }
// }

// #include "qtensor.h"
// #include "tensor.h"

void linear_forward_q8(
    const QTensor *x_q,      // (B x K)
    const QTensor *W_q,      // (N x K)  TRANSPOSED
    const Tensor  *bias,
    Tensor *out              // (B x N)
)
{
    int B = x_q->rows;
    int K = x_q->cols;
    int N = W_q->rows;   // because transposed

    if (out->rows != B || out->cols != N) {
        fprintf(stderr, "linear_forward_q8 shape mismatch\n");
        exit(1);
    }

    int32_t *accum = (int32_t*)malloc(sizeof(int32_t) * B * N);
    if (!accum) {
        fprintf(stderr, "malloc failed in linear_forward_q8\n");
        exit(1);
    }

    // -------- INT8 MATMUL (AVX2) --------
    matmul_q8_avx2(
        x_q->data,
        W_q->data,
        accum,
        B,      // M
        N,      // N
        K       // K
    );

    float total_scale = x_q->scale * W_q->scale;

    // -------- Dequantize + Bias --------
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {

            float val = accum[i*N + j] * total_scale;

            if (bias)
                val += bias->data[j];

            out->data[i*N + j] = val;
        }
    }

    free(accum);
}
