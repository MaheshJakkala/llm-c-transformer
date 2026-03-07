// activations.c
#include "activations.h"
// #include <math.h>

// void gelu_inplace(Tensor *t) {
//     if (!t) {
//         fprintf(stderr, "gelu_inplace: null tensor\n");
//         return;
//     }

//     // Reuse forward implementation
//     Tensor *tmp = tensor_create(t->rows, t->cols);
//     gelu_forward(t, tmp);

//     // Copy back
//     size_t n = t->rows * t->cols;
//     for (size_t i = 0; i < n; ++i) {
//         t->data[i] = tmp->data[i];
//     }

//     tensor_free(tmp);
// }
void gelu_inplace(Tensor *t) {
    if (!t || !t->data){
        printf("Error in gelu_inplace ");
        return;
    }

    size_t n = t->rows * t->cols;
    for (size_t i = 0; i < n; ++i) {
        float x = t->data[i];
        t->data[i] = 0.5f * x * (1.0f + tanhf(
            0.79788456f * (x + 0.044715f * x * x * x)
        ));
    }
}

// Gaussian Error Linear Unit (GELU) activation
void gelu_forward(const Tensor *x, Tensor *y) {
    if (x->rows != y->rows || x->cols != y->cols) {
        fprintf(stderr, "gelu_forward: shape mismatch\n");
        return;
    }
    size_t n = x->rows * x->cols;
    for (size_t i = 0; i < n; ++i) {
        float xi = x->data[i];
        y->data[i] = 0.5f * xi * (1.0f + tanhf(0.79788456f * (xi + 0.044715f * xi * xi * xi)));
    }
}

void gelu_backward(const Tensor *x, const Tensor *grad_out, Tensor *grad_in) {
    size_t n = x->rows * x->cols;

    for (size_t i = 0; i < n; ++i) {
        float xi = x->data[i];

        float t = 0.79788456f * (xi + 0.044715f * xi * xi * xi);
        float tanh_t = tanhf(t);
        float sech2 = 1.0f - tanh_t * tanh_t;

        float dt_dx = 0.79788456f * (1.0f + 3.0f * 0.044715f * xi * xi);

        float dgelu_dx =
            0.5f * (1.0f + tanh_t) +
            0.5f * xi * sech2 * dt_dx;

        grad_in->data[i] = grad_out->data[i] * dgelu_dx;
    }
}
