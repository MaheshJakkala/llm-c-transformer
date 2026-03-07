/*
 * lm_config.h — Causal Language Model Configuration
 *
 * This is a character-level causal transformer language model:
 * given a context window of LM_CTX characters, predict the next character.
 * Vocabulary = 256 (all ASCII bytes). This is exactly what GPT does,
 * scaled down to run on CPU in seconds.
 */
#ifndef LM_CONFIG_H
#define LM_CONFIG_H

#include <immintrin.h>
#include <stdint.h>

/* ── Vocabulary ── */
#define LM_VOCAB        256     /* byte-level: all ASCII values        */

/* ── Architecture ── */
#define LM_CTX          32      /* context window (tokens)             */
#define LM_HIDDEN       128     /* embedding / attention dimension      */
#define LM_FFN          256     /* FFN intermediate (2 × LM_HIDDEN)    */

/* ── Training ── */
#define LM_EPOCHS       25
#define LM_LR           1e-3f
#define LM_GRAD_CLIP    1.0f
#define LM_BATCH        64      /* number of (context, target) pairs per step */

/* ── Generation ── */
#define LM_GEN_LEN      200     /* characters to generate at eval time */

/* ── Sentinel ── */
#define LM_PAD          -1

#endif /* LM_CONFIG_H */
