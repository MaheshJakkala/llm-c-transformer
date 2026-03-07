#ifndef CONFIG_H
#define CONFIG_H

#include <immintrin.h>
#include <stdint.h>

/* ── Dataset ── */
#define NUM_SAMPLES     200
#define MAX_SEQ_LEN     64
#define NUM_CLASSES     9        /* CoNLL-2003 BIO: O B-PER I-PER B-ORG I-ORG B-LOC I-LOC B-MISC I-MISC */

#define TRAIN_SAMPLES   200
#define VAL_SAMPLES     50

/* ── Training ── */
#define EPOCHS          30
#define LR              3e-4f
#define GRAD_CLIP       1.0f

/* ── Architecture ── */
#define HIDDEN_SIZE     256
#define FFN_HIDDEN      512     /* 2 × HIDDEN_SIZE */
#define VOCAB_SIZE      4096

/* ── Sentinel ── */
#define PAD             -1

#endif /* CONFIG_H */
