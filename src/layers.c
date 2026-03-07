#include "tensor.h"
#include "ops.h"


// backward
// void linear_backward(const Tensor *x, const Tensor *grad_out, Tensor *grad_W, Tensor *grad_b){
//     // grad_W = x^T * grad_out
//     Tensor *xT = tensor_create(x->cols, x->rows);
//     for(int i=0;i<x->rows;i++)
//         for(int j=0;j<x->cols;j++)
//             xT->data[j*x->rows + i] = x->data[i*x->cols + j];
//     matmul_naive(xT, grad_out, grad_W);
//     tensor_free(xT);

//     // grad_b = sum over batch
//     for(int j=0;j<grad_b->cols;j++){
//         grad_b->data[j]=0.0f;
//         for(int i=0;i<grad_out->rows;i++)
//             grad_b->data[j] += grad_out->data[i*grad_out->cols + j];
//     }
// }
