// src/tokenizer.h
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "tensor.h"
#include<string.h>
#include<math.h>
// #include<utils.h>

#define VOCAB_SIZE 256  // 1 byte = 256 possible tokens
#define PAD_ID 0


typedef struct {
    int vocab_size;
} Tokenizer;

Tokenizer *tokenizer_create();
void tokenizer_free(Tokenizer *t);
void encode(const char *text, int *output, int max_len, int *seq_len);
// In src/utils.h or similar header file
unsigned int simple_hash(const char* str, unsigned int vocab_size);

void encode_word(const char *text, int *output, int max_len, int *seq_len);

void encode_word_with_tokens(
    const char *text,
    int *output,
    char tokens[][32],   // store words
    int max_len,
    int *seq_len
);


#endif
