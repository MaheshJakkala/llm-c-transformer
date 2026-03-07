# Makefile — Transformer LLM in C
# Targets:
#   make all        → builds lm, train, bench
#   make lm         → causal language model (next-token prediction)
#   make train      → NER fine-tuning demo
#   make bench      → FP32 vs INT8-AVX2 inference benchmark

CC      = gcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -Wno-unused-parameter
CFLAGS += -fopenmp
CFLAGS += -mavx2 -mfma
LDFLAGS = -lm -fopenmp
IFLAGS  = -Isrc -Idata

SRC_COMMON = \
    src/tensor.c src/qtensor.c src/ops.c src/layers.c \
    src/layernorm.c src/activations.c src/linear.c \
    src/attention.c src/transformer_block.c src/loss.c \
    src/optimizer.c src/ffn.c src/tokenizer.c \
    src/embedding.c src/evaluation.c src/arena.c \
    data/data.c data/cJSON.c

SRC_LM    = $(SRC_COMMON) src/lm_train.c
SRC_TRAIN = $(SRC_COMMON) src/main.c
SRC_BENCH = $(SRC_COMMON) src/bench.c

.PHONY: all lm train bench clean

all: lm train bench

lm:
	$(CC) $(CFLAGS) $(IFLAGS) -o lm $(SRC_LM) $(LDFLAGS)

train:
	$(CC) $(CFLAGS) $(IFLAGS) -o train $(SRC_TRAIN) $(LDFLAGS)

bench:
	$(CC) $(CFLAGS) $(IFLAGS) -o bench $(SRC_BENCH) $(LDFLAGS)

clean:
	rm -f lm train bench
