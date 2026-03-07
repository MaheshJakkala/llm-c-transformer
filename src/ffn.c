// ffn.c
#include "ffn.h"

// input: seq x d_model
// W1: d_model x hidden  => temp1 = input * W1  => seq x hidden
// GELU(temp1)
// out = temp1 * W2 + b2  => seq x d_model
void ffn_forward(const Tensor *input,
                 const Tensor *W1, const Tensor *b1,
                 const Tensor *W2, const Tensor *b2,
                 Tensor *out, Tensor *temp1) {
    // temp1 = input * W1 + b1
    linear_forward(input, W1, b1, temp1);

    // activation in-place on temp1 (GELU)
    gelu_inplace(temp1);

    // out = temp1 * W2 + b2
    linear_forward(temp1, W2, b2, out);
}

// #include "transformer_block_backward.h"
void ffn_backward(
    const Tensor *attn_out,   // INPUT to FFN
    const Tensor *ff1,        // GELU output
    const Tensor *grad_out,   // dL/d(out)
    const Tensor *W2,
    const Tensor *W1,
    Tensor *grad_W2,
    Tensor *grad_b2,
    Tensor *grad_W1,
    Tensor *grad_b1,
    Tensor *grad_attn_out     // OUTPUT gradient
) {
    if (!ff1) {
    fprintf(stderr, "ffn_backward: ff1 is NULL\n");
    exit(1);
}

    // dL/d(ff1)
    Tensor *grad_ff1 = tensor_create(ff1->rows, ff1->cols);
    
    // printf("ff1 in ff1 backward: ");
    // tensor_print(ff1);

    linear_backward(
        ff1, W2, grad_out,
        grad_ff1, grad_W2, grad_b2
    );

    // GELU backward
    Tensor *grad_ff1_pre = tensor_create(ff1->rows, ff1->cols);
    gelu_backward(ff1, grad_ff1, grad_ff1_pre);
    tensor_free(grad_ff1);

    // dL/d(attn_out)
    linear_backward(
        attn_out, W1, grad_ff1_pre,
        grad_attn_out, grad_W1, grad_b1
    );

    tensor_free(grad_ff1_pre);
}

// #include "ffn.h"

void ffn_forward_q8(
    const QTensor *input_q,
    const QTensor *W1_q,
    const Tensor  *b1,
    const QTensor *W2_q,
    const Tensor  *b2,
    Tensor *out,
    Tensor *temp1
) {
    linear_forward_q8(input_q, W1_q, b1, temp1);
    gelu_inplace(temp1);

    QTensor *temp1_q = quantize_tensor(temp1);
    linear_forward_q8(temp1_q, W2_q, b2, out);
    free_qtensor(temp1_q);
}
