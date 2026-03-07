#include "data.h"

/* Entities */
static const char *PER[] = {
    "John", "Alice", "Ravi", "Priya",
    "Sundar Pichai", "Elon Musk", "Satya Nadella"
};

static const char *ORG[] = {
    "Google", "Microsoft", "Amazon",
    "OpenAI", "Apple", "Meta", "Tesla"
};

static const char *LOC[] = {
    "India", "USA", "London",
    "Paris", "Berlin", "Tokyo"
};

/* Templates WITH metadata */
static Template TEMPLATES[] = {
    { "%s is CEO of %s",        PER_ORG },
    { "%s joined %s",           PER_ORG },
    { "%s works at %s in %s",   PER_ORG_LOC },
    { "%s moved to %s",         PER_LOC },
    { "%s opened office in %s", PER_ORG }
};



// void apply_bio_labels(
//     char tokens[MAX_SEQ_LEN][32],
//     int token_count,
//     const char *entity,
//     int B_TAG,
//     int I_TAG,
//     int labels[MAX_SEQ_LEN]
// ) {
//     char ent_copy[64];
//     strcpy(ent_copy, entity);

//     char *ent_tokens[8];
//     int ent_len = 0;

//     char *tok = strtok(ent_copy, " ");
//     while (tok && ent_len < 8) {
//         ent_tokens[ent_len++] = tok;
//         tok = strtok(NULL, " ");
//     }

//     for (int i = 0; i <= token_count - ent_len; i++) {
//         int match = 1;
//         for (int j = 0; j < ent_len; j++) {
//             if (strcmp(tokens[i + j], ent_tokens[j]) != 0) {
//                 match = 0;
//                 break;
//             }
//         }
//         if (match) {
//             labels[i] = B_TAG;
//             for (int j = 1; j < ent_len; j++)
//                 labels[i + j] = I_TAG;
//             return;
//         }
//     }
// }
static void assign_labels(
    const char *sentence,
    int *labels,
    int max_len,
    TemplateType type
){
    int input_ids[MAX_SEQ_LEN];
    int seq_len;

    encode_word(sentence, input_ids, max_len, &seq_len);

    for (int i = 0; i < seq_len; i++)
        labels[i] = O_TAG;

    if (type == PER_ONLY || type == PER_ORG || type == PER_ORG_LOC) {
        labels[0] = B_PER;
    }

    if (type == PER_ORG || type == PER_ORG_LOC) {
        labels[2] = B_ORG;   // safe because templates fixed
    }

    if (type == PER_ORG_LOC) {
        labels[4] = B_LOC;
    }

    for (int i = seq_len; i < max_len; i++)
        labels[i] = -1;
}


int tokenize(char *sentence, char tokens[MAX_SEQ_LEN][32]) {
    int count = 0;
    char *tok = strtok(sentence, " ");
    while (tok && count < MAX_SEQ_LEN) {
        strcpy(tokens[count++], tok);
        tok = strtok(NULL, " ");
    }
    return count;
}


// void generate_sample(
//     char *out_text,
//     int labels[MAX_SEQ_LEN]
// ) {
//     const char *per = PER[rand() % 7];
//     const char *org = ORG[rand() % 7];
//     const char *loc = LOC[rand() % 6];
//     const char *tpl = TEMPLATES[rand() % 5];

//     // snprintf(out_text, 128, tpl, per, org, loc);
//     int t = rand() % 5;

//     if (t == 0)
//     {
//         snprintf(out_text, 128, TEMPLATES[t], per, org, loc);
//     } else {
//         snprintf(out_text, 128, TEMPLATES[t], per, org);
//     }


//     char temp[128];
//     strcpy(temp, out_text);

//     char tokens[MAX_SEQ_LEN][32];
//     int token_count = tokenize(temp, tokens);

//     for (int i = 0; i < MAX_SEQ_LEN; i++)
//         labels[i] = O_TAG;

//     apply_bio_labels(tokens, token_count, per, B_PER, I_PER, labels);
//     apply_bio_labels(tokens, token_count, org, B_ORG, I_ORG, labels);
//     apply_bio_labels(tokens, token_count, loc, B_LOC, I_LOC, labels);
// }

static void generate_sample(
    char *out_text,
    int labels[MAX_SEQ_LEN]
){
    const char *per = PER[rand() % 7];
    const char *org = ORG[rand() % 7];
    const char *loc = LOC[rand() % 6];

    int t = rand() % NUM_TEMPLATES;
    Template tpl = TEMPLATES[t];

    if (tpl.type == PER_ORG_LOC)
        snprintf(out_text, 128, tpl.fmt, per, org, loc);
    else if (tpl.type == PER_LOC)
        snprintf(out_text, 128, tpl.fmt, per, loc);
    else
        snprintf(out_text, 128, tpl.fmt, per, org);

    assign_labels(out_text, labels, MAX_SEQ_LEN, tpl.type);
}


void generate_dataset(
    char train_texts[TRAIN_SAMPLES][128],
    int  train_labels[TRAIN_SAMPLES][MAX_SEQ_LEN],
    char val_texts[VAL_SAMPLES][128],
    int  val_labels[VAL_SAMPLES][MAX_SEQ_LEN]
) {
    // srand(time(NULL));

    for (int i = 0; i < TRAIN_SAMPLES; i++)
        generate_sample(train_texts[i], train_labels[i]);

    for (int i = 0; i < VAL_SAMPLES; i++)
        generate_sample(val_texts[i], val_labels[i]);
}



int conll_label_to_id(const char *tag) {
    if (strcmp(tag, "O") == 0) return O_TAG;

    if (strcmp(tag, "B-PER") == 0) return B_PER;
    if (strcmp(tag, "I-PER") == 0) return I_PER;

    if (strcmp(tag, "B-ORG") == 0) return B_ORG;
    if (strcmp(tag, "I-ORG") == 0) return I_ORG;

    if (strcmp(tag, "B-LOC") == 0) return B_LOC;
    if (strcmp(tag, "I-LOC") == 0) return I_LOC;

    if (strcmp(tag, "B-MISC") == 0) return B_MISC;
    if (strcmp(tag, "I-MISC") == 0) return I_MISC;

    return O_TAG; // fallback
}

// void load_conll_dataset(
//     const char *filename,
//     char texts[][128],
//     int labels[][MAX_SEQ_LEN],
//     int max_samples,
//     int *num_loaded
// ){
//     FILE *fp = fopen(filename, "r");
//     if (!fp) {
//         perror("fopen");
//         exit(1);
//     }

//     char line[256];
//     char sentence[128] = "";
//     int label_buf[MAX_SEQ_LEN];
//     int tok_count = 0;
//     int sample = 0;

//     while (fgets(line, sizeof(line), fp)) {
//         // Sentence boundary
//         if (line[0] == '\n' || line[0] == '\r') {
//             if (tok_count > 0 && sample < max_samples) {
//                 strcpy(texts[sample], sentence);

//                 for (int i = 0; i < tok_count; i++)
//                     labels[sample][i] = label_buf[i];

//                 for (int i = tok_count; i < MAX_SEQ_LEN; i++)
//                     labels[sample][i] = -1;

//                 sample++;
//                 sentence[0] = '\0';
//                 tok_count = 0;
//             }
//             continue;
//         }

//         char token[64], pos[32], chunk[32], ner[32];
//         sscanf(line, "%s %s %s %s", token, pos, chunk, ner);

//         if (tok_count < MAX_SEQ_LEN) {
//             if (tok_count > 0) strcat(sentence, " ");
//             strcat(sentence, token);

//             label_buf[tok_count++] = conll_label_to_id(ner);
//         }
//     }

//     fclose(fp);
//     *num_loaded = sample;
// }
/* join tokens into a single sentence */
static void join_tokens(cJSON *tokens, char *out) {
    out[0] = '\0';
    int n = cJSON_GetArraySize(tokens);

    for (int i = 0; i < n; i++) {
        strcat(out, cJSON_GetArrayItem(tokens, i)->valuestring);
        if (i != n - 1) strcat(out, " ");
    }
}

void load_conll_dataset(
    const char *filename,
    char texts[][128],
    int  labels[][MAX_SEQ_LEN],
    int  max_samples,
    int *num_loaded
){
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("fopen");
        exit(1);
    }

    char line[4096];
    int sample = 0;

    while (fgets(line, sizeof(line), fp) && sample < max_samples) {

        cJSON *root = cJSON_Parse(line);
        if (!root) continue;

        cJSON *tokens = cJSON_GetObjectItem(root, "tokens");
        cJSON *tags   = cJSON_GetObjectItem(root, "tags");

        if (!tokens || !tags) {
            cJSON_Delete(root);
            continue;
        }

        /* sentence */
        join_tokens(tokens, texts[sample]);

        int seq_len = cJSON_GetArraySize(tokens);
        if (seq_len > MAX_SEQ_LEN)
            seq_len = MAX_SEQ_LEN;

        /* labels */
        for (int i = 0; i < seq_len; i++) {
            labels[sample][i] =
                cJSON_GetArrayItem(tags, i)->valueint;
        }

        /* padding */
        for (int i = seq_len; i < MAX_SEQ_LEN; i++)
            labels[sample][i] = PAD;

        sample++;
        cJSON_Delete(root);
    }

    fclose(fp);
    *num_loaded = sample;
}


