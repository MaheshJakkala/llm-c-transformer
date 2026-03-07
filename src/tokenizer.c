// src/tokenizer.c
#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Tokenizer *tokenizer_create() {
    Tokenizer *t = (Tokenizer*)malloc(sizeof(Tokenizer));
    t->vocab_size = VOCAB_SIZE;
    return t;
}

void tokenizer_free(Tokenizer *t) {
    if (t) free(t);
}

// Encode text as sequence of bytes (ints 0-255)
void encode(const char *text, int *output, int max_len, int *seq_len) {
    int len = strlen(text);
    if (len > max_len) len = max_len;

    for (int i = 0; i < len; ++i)
        output[i] = (unsigned char)text[i];  // cast char to 0-255

    *seq_len = len;
}

unsigned int simple_hash(const char* str, unsigned int vocab_size) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; // djb2 hash algorithm
    }
    return hash % vocab_size;
}

//worl level tokenization
void encode_word(const char *text, int *output, int max_len, int *seq_len){
    char buf[1024];
    strncpy(buf, text, sizeof(buf));
    buf[sizeof(buf)-1] = '\0';

    int count = 0;
    char *tok = strtok(buf, " ");
    while (tok && count < max_len) {
        output[count++] = simple_hash(tok, VOCAB_SIZE);
        tok = strtok(NULL, " ");
    }
    *seq_len = count;
}

void encode_word_with_tokens(
    const char *text,
    int *output,
    char tokens[][32],
    int max_len,
    int *seq_len
){
    char buf[1024];
    strncpy(buf, text, sizeof(buf));
    buf[sizeof(buf)-1] = '\0';

    int count = 0;
    char *tok = strtok(buf, " ");
    while (tok && count < max_len) {
        output[count] = simple_hash(tok, VOCAB_SIZE);

        // save token text (truncate if long)
        strncpy(tokens[count], tok, 31);
        tokens[count][31] = '\0';

        count++;
        tok = strtok(NULL, " ");
    }
    *seq_len = count;
}

