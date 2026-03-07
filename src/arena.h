/*
 * arena.h — Tensor Memory Arena (Bump Allocator)
 *
 * Instead of calling malloc/free for every activation tensor on every
 * training step, we pre-allocate one large contiguous block and hand
 * out slices from it. Resetting the arena is O(1) — just reset the
 * offset pointer. This eliminates allocator overhead in the training
 * hot path and improves cache locality for activation tensors.
 *
 * Usage:
 *   TensorArena *a = arena_create(64 * 1024 * 1024);  // 64 MB pool
 *   Tensor *x = arena_tensor(a, rows, cols);           // zero-copy alloc
 *   arena_reset(a);                                    // free all at once
 *   arena_free(a);
 */
#ifndef ARENA_H
#define ARENA_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "tensor.h"

typedef struct {
    float  *base;       /* contiguous float pool                   */
    size_t  capacity;   /* total floats available                  */
    size_t  used;       /* floats handed out so far                */
    size_t  peak;       /* high-water mark (for reporting)         */
    int     n_allocs;   /* number of tensor_create calls saved     */

    /* header pool: Tensor structs, separate from float data */
    Tensor *header_pool;
    int     header_cap;
    int     header_used;
} TensorArena;

/* Create an arena that can hold at most `float_capacity` floats */
TensorArena *arena_create(size_t float_capacity);

/* Allocate a Tensor from the arena (data + header both from pool) */
Tensor *arena_tensor(TensorArena *a, int rows, int cols);

/* Reset the arena — O(1), reuse all memory without free/malloc */
void arena_reset(TensorArena *a);

/* Print peak usage stats */
void arena_report(const TensorArena *a);

/* Free the arena and all pooled memory */
void arena_free(TensorArena *a);

#endif /* ARENA_H */
