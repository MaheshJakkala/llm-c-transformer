/*
 * arena.c — Tensor Memory Arena Implementation
 *
 * Design:
 *  - Two contiguous pools: one for float data, one for Tensor headers.
 *  - arena_tensor() bumps the pointer forward; no free per-tensor.
 *  - arena_reset() sets both pointers back to zero in O(1).
 *  - This removes thousands of malloc/free calls per epoch from the
 *    training hot path, reducing allocator overhead and heap fragmentation.
 */
#include "arena.h"

TensorArena *arena_create(size_t float_capacity) {
    TensorArena *a = (TensorArena *)malloc(sizeof(TensorArena));
    if (!a) { fprintf(stderr, "arena_create: malloc failed\n"); exit(1); }

    a->base     = (float *)malloc(float_capacity * sizeof(float));
    if (!a->base) { fprintf(stderr, "arena_create: float pool malloc failed\n"); exit(1); }

    /* Pre-allocate header pool (max 256 Tensor structs per arena reset) */
    a->header_cap  = 256;
    a->header_pool = (Tensor *)malloc(a->header_cap * sizeof(Tensor));
    if (!a->header_pool) { fprintf(stderr, "arena_create: header pool malloc failed\n"); exit(1); }

    a->capacity     = float_capacity;
    a->used         = 0;
    a->peak         = 0;
    a->n_allocs     = 0;
    a->header_used  = 0;

    return a;
}

Tensor *arena_tensor(TensorArena *a, int rows, int cols) {
    size_t n = (size_t)(rows * cols);

    if (a->used + n > a->capacity) {
        fprintf(stderr,
            "arena_tensor: out of space (need %zu, have %zu, used %zu)\n",
            n, a->capacity, a->used);
        exit(1);
    }
    if (a->header_used >= a->header_cap) {
        fprintf(stderr, "arena_tensor: header pool exhausted\n");
        exit(1);
    }

    /* Carve Tensor header from header pool */
    Tensor *t       = &a->header_pool[a->header_used++];
    t->rows         = rows;
    t->cols         = cols;
    t->data         = a->base + a->used;

    /* Zero-initialise the data region */
    memset(t->data, 0, n * sizeof(float));

    a->used    += n;
    a->n_allocs++;
    if (a->used > a->peak) a->peak = a->used;

    return t;
}

void arena_reset(TensorArena *a) {
    a->used        = 0;
    a->header_used = 0;
    /* Data is NOT zeroed here — arena_tensor() zeros on allocation */
}

void arena_report(const TensorArena *a) {
    printf("  Arena peak usage : %.2f MB / %.2f MB (%.0f%%)\n",
           a->peak  * sizeof(float) / (1024.0 * 1024.0),
           a->capacity * sizeof(float) / (1024.0 * 1024.0),
           100.0 * a->peak / a->capacity);
    printf("  Malloc calls saved (per epoch): %d\n", a->n_allocs);
}

void arena_free(TensorArena *a) {
    free(a->base);
    free(a->header_pool);
    free(a);
}
