#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include "../src/config.h"
#include "../src/tokenizer.h"   // <-- for encode_word
// #include "src/tokenizer.h"

#pragma once
#define O_TAG   0
#define B_PER   1
#define I_PER   2
#define B_ORG   3
#define I_ORG   4
#define B_LOC   5
#define I_LOC   6
#define B_MISC  7
#define I_MISC  8
#define NUM_TEMPLATES (sizeof(TEMPLATES)/sizeof(TEMPLATES[0]))

typedef enum {
    PER_ONLY,
    PER_ORG,
    PER_LOC,
    PER_ORG_LOC
} TemplateType;

typedef struct {
    const char *fmt;
    TemplateType type;
} Template;



// extern const char *PER[];
// extern const char *ORG[];
// extern const char *LOC[];
// extern const char *TEMPLATES[];

void apply_bio_labels(
    char tokens[MAX_SEQ_LEN][32],
    int token_count,
    const char *entity,
    int B_TAG,
    int I_TAG,
    int labels[MAX_SEQ_LEN]
);
int tokenize(char *sentence, char tokens[MAX_SEQ_LEN][32]);
// void generate_sample(
//     char *out_text,
//     int labels[MAX_SEQ_LEN]
// );
void generate_dataset(
    char train_texts[TRAIN_SAMPLES][128],
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
    char val_texts[VAL_SAMPLES][128],
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
);

void load_conll_dataset(
    const char *filename,
    char texts[][128],
    int labels[][MAX_SEQ_LEN],
    int max_samples,
    int *num_loaded
);