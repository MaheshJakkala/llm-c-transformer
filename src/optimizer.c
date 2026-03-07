#include "optimizer.h"
#include <math.h>
#include <stdlib.h>

AdamState* adam_create(float lr, float beta1, float beta2, float epsilon, int rows, int cols){
    AdamState *s = malloc(sizeof(AdamState));
    s->lr = lr; s->beta1 = beta1; s->beta2 = beta2; s->epsilon = epsilon;
    s->m = tensor_create(rows, cols);
    s->v = tensor_create(rows, cols);
    s->t = 0;
    return s;
}

void adam_step(Tensor *param, const Tensor *grad, AdamState *s){
    s->t += 1;
    int N = param->rows * param->cols;
    float b1t = 1.0f - powf(s->beta1, s->t);
    float b2t = 1.0f - powf(s->beta2, s->t);

    for(int i=0;i<N;i++){
        s->m->data[i] = s->beta1 * s->m->data[i] + (1.0f - s->beta1) * grad->data[i];
        s->v->data[i] = s->beta2 * s->v->data[i] + (1.0f - s->beta2) * grad->data[i] * grad->data[i];
        float m_hat = s->m->data[i] / b1t;
        float v_hat = s->v->data[i] / b2t;
        param->data[i] -= s->lr * m_hat / (sqrtf(v_hat) + s->epsilon);
    }
}

void adam_free(AdamState *s){
    tensor_free(s->m);
    tensor_free(s->v);
    free(s);
}
