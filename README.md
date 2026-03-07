# Transformer-Based LLM from Scratch in C

<div align="center">

![Language](https://img.shields.io/badge/Language-C11-blue?logo=c)
![SIMD](https://img.shields.io/badge/SIMD-AVX2%2FFMA-orange)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Designed and implemented a transformer-based LLM from scratch in C — custom tensor library, INT8 post-training quantization with AVX2 SIMD, memory-efficient training pipeline (tensor arena allocator), and benchmarked performance improvements over baseline PyTorch/NumPy implementations.**

</div>

---

## What This Project Is

This project implements the **full transformer stack in pure C** — no ML framework, no BLAS, no external dependencies beyond libc.

| Component | File | Task |
|-----------|------|------|
| **Causal Language Model** | `src/lm_train.c` | Next-token prediction on raw text. Given 32 chars, predict the 33rd. This is the LLM component. |
| **NER Fine-Tuning** | `src/main.c` | Applies the same transformer to CoNLL-2003 named-entity tagging as a downstream demo. |
| **Inference Benchmark** | `src/bench.c` | FP32 vs INT8-AVX2 vs NumPy/PyTorch baseline — real measurements. |

---

## Architecture

```
Input (chars)
     │
     ▼
Embedding Table (256 × 128)    ← byte-level vocab, learned
     │
     ▼
┌──────────────────────────────────┐
│  Transformer Block                │
│                                   │
│  Q = x·Wq  K = x·Wk  V = x·Wv   │  linear projections (128×128)
│  scores = QKᵀ / √d               │  scaled dot-product attention
│  attn = softmax(scores)           │
│  out  = attn·V·Wo                 │  output projection
│  x    = LayerNorm(x + out)        │  residual connection
│                                   │
│  h = GELU(x·W1 + b1)             │  FFN: 128 → 256
│  out = h·W2 + b2                  │  FFN: 256 → 128
│  x   = LayerNorm(x + out)         │  residual connection
└──────────────────────────────────┘
     │
     ▼
LM Head: Linear(128 → 256)     ← predict next character
     │
     ▼
Cross-Entropy Loss
     │
     ▼
Full Backprop (hand-derived gradients through every layer)
     │
     ▼
Adam optimizer  +  Gradient Clipping (L2, max=1.0)
```

---

## Results — All Numbers Measured on This Machine

### Causal Language Model Training

| Epoch | Loss   | Perplexity |
|-------|--------|------------|
| 1     | 4.4861 | 88.78      |
| 5     | 2.9843 | 19.77      |
| 15    | 2.8111 | 16.63      |
| 21    | 2.7789 | 16.10      |
| **25**| **2.7284** | **15.31** |

Total training: **20.04 s** · **801 ms/epoch** · **11,200 malloc calls eliminated per epoch** (arena)

---

### Inference Benchmark: C vs PyTorch/NumPy Baseline

| Seq | NumPy/PyTorch (ms) | C FP32 (ms) | C INT8-AVX2 (ms) | INT8 vs PyTorch |
|-----|--------------------|-------------|------------------|-----------------|
| 8   | 1.264              | 5.002       | **0.170**        | **7.4×**        |
| 16  | 2.355              | 10.847      | **0.275**        | **8.6×**        |
| 32  | 4.242              | 20.893      | **0.651**        | **6.5×**        |
| 64  | 8.124              | 42.070      | **4.432**        | **1.8×**        |

> NumPy uses OpenBLAS for matmul which is faster than naive C FP32. The C INT8-AVX2 path beats PyTorch by processing 32 INT8 multiplies per SIMD instruction and eliminating Python/framework overhead entirely.

---

### MatMul Micro-Benchmark (M=N=K=256)

| Backend           | Time (ms) | Throughput       |
|-------------------|-----------|------------------|
| NumPy/PyTorch CPU | 0.376 ms  | 89.33 GFLOP/s    |
| C FP32 Naive      | 3.089 ms  | 10.86 GFLOP/s    |
| **C INT8-AVX2**   | **0.988 ms** | **33.97 GOPS/s** |

INT8-AVX2 is **3.13× faster than C FP32** at the matmul level.

---

### Memory Footprint

| Precision | Transformer Weights | Reduction |
|-----------|---------------------|-----------|
| FP32      | 2.01 MB             | baseline  |
| **INT8**  | **0.50 MB**         | **4.00×** |

---

### Memory Arena — Training Allocator

| Method   | malloc/free per step | Behaviour |
|----------|---------------------|-----------|
| Standard | 512 calls           | heap fragmentation, cold cache |
| **Arena**| **0 calls** (pool reset in O(1)) | **11,200 calls eliminated/epoch** |

---

## Plots

<table>
<tr>
<td align="center"><img src="results/plots/lm_training_loss.png"/><br><b>LM Loss & Perplexity</b></td>
<td align="center"><img src="results/plots/c_vs_pytorch.png"/><br><b>C vs PyTorch Latency</b></td>
</tr>
<tr>
<td align="center"><img src="results/plots/matmul_benchmark.png"/><br><b>MatMul Micro-Benchmark</b></td>
<td align="center"><img src="results/plots/memory_footprint.png"/><br><b>Weight Memory (4× saving)</b></td>
</tr>
<tr>
<td align="center"><img src="results/plots/throughput.png"/><br><b>Throughput vs PyTorch</b></td>
<td align="center"><img src="results/plots/arena_memory.png"/><br><b>Arena vs Standard Allocator</b></td>
</tr>
</table>

---

## Repository Structure

```
llm-c-transformer/
├── src/
│   ├── lm_train.c          ← ★ Causal LM: next-token prediction (LLM task)
│   ├── arena.c / arena.h   ← ★ Tensor memory pool (memory-efficient training)
│   ├── lm_config.h         ←   LM hyperparameters
│   ├── bench.c             ←   Benchmark binary (FP32 vs INT8 vs NumPy)
│   ├── main.c              ←   NER downstream task demo
│   ├── tensor.c/h          ←   Custom tensor library
│   ├── qtensor.c/h         ←   INT8 quantization + AVX2 matmul
│   ├── attention.c/h       ←   Scaled dot-product attention + backward pass
│   ├── ffn.c/h             ←   Feed-forward block + backward pass
│   ├── transformer_block.c ←   Full block: Attn + LN + FFN
│   ├── linear.c/h          ←   Linear layer (FP32 + INT8-AVX2 path)
│   ├── layernorm.c/h       ←   Layer normalisation
│   ├── activations.c/h     ←   GELU (fast tanh approx) + backward
│   ├── loss.c/h            ←   Cross-entropy (log-sum-exp stable)
│   ├── optimizer.c/h       ←   Adam + gradient clipping
│   ├── embedding.c/h       ←   Embedding lookup + backward
│   └── tokenizer.c/h       ←   Byte tokeniser (vocab=256)
├── data/
│   ├── corpus.txt          ←   Text corpus for LM training
│   └── conll2003/          ←   CoNLL-2003 NER dataset
├── results/
│   ├── plots/              ←   7 benchmark plots (real measurements)
│   └── metrics/            ←   CSV files + training/benchmark logs
├── notebooks/
│   └── LLMFromScratch.ipynb  ← PyTorch baseline (reference implementation)
├── scripts/
│   └── plot_results.py
└── Makefile
```

---

## Build & Run

```bash
# Ubuntu / Debian — GCC ≥ 10 required
sudo apt install gcc libgomp1 make

make all          # builds: lm, train, bench

./lm              # train the language model
./bench           # run inference benchmark vs NumPy/PyTorch
./train           # NER fine-tuning demo

python3 scripts/plot_results.py   # regenerate all plots
```

---

## Key Implementation Details

### Custom Tensor Library (`src/tensor.c`)
```c
typedef struct { int rows, cols; float *data; } Tensor;
Tensor *tensor_create(int rows, int cols);   // heap alloc + zero-init
void    tensor_free(Tensor *t);
```

### Memory-Efficient Training: Tensor Arena (`src/arena.c`)
```c
TensorArena *arena = arena_create(64 * 1024 * 1024);  // 64 MB pool once

// Every training step — activations come from the pool, no malloc:
Tensor *x      = arena_tensor(arena, ctx, hidden);
Tensor *out    = arena_tensor(arena, ctx, hidden);
Tensor *logits = arena_tensor(arena, ctx, vocab);

arena_reset(arena);  // O(1) reset — reuse all memory next step
```
Eliminates **11,200 malloc/free calls per epoch** in LM training. Pool stays warm in CPU cache, improving locality for activations reused in the backward pass.

### INT8 Quantization + AVX2 MatMul (`src/qtensor.c`, `src/tensor.c`)
```c
// Symmetric per-tensor quantization: scale = max_abs / 127
QTensor *quantize_weight_transpose(const Tensor *W);

// 32 INT8 multiply-accumulates per SIMD instruction
void matmul_q8_avx2(const int8_t *A, const int8_t *B, int32_t *C,
                    int M, int N, int K);
// _mm256_maddubs_epi16 + _mm256_madd_epi16
// 3.13× faster than scalar FP32, 4.00× less memory
```

### Full Backpropagation
```
cross_entropy_grad
  → linear_backward (W_lm)
    → ffn_backward (W2, W1)
      → attention_backward (Wo, Wv, Wk, Wq)
        → embedding_backward
```
All gradients hand-derived. Adam updates with L2 gradient clipping applied after each step.

---

## vs PyTorch Baseline

[`notebooks/LLMFromScratch.ipynb`](notebooks/LLMFromScratch.ipynb) contains the equivalent PyTorch implementation used as the reference and baseline.

| Metric | PyTorch/NumPy CPU | C INT8-AVX2 |
|--------|-------------------|-------------|
| Weight memory | 2.01 MB | **0.50 MB (4×)** |
| Inference latency (seq=16) | 2.355 ms | **0.275 ms (8.6×)** |
| Training allocator | malloc/free per step | **Arena pool (O(1) reset)** |
| Dependencies | PyTorch, NumPy, BLAS | libc, libm, libgomp only |

---

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dettmers et al. (2022). [LLM.int8()](https://arxiv.org/abs/2208.07339)
- Karpathy, A. [llm.c](https://github.com/karpathy/llm.c) — inspiration for this project
- Sang & De Meulder (2003). [CoNLL-2003 NER Shared Task](https://arxiv.org/abs/cs/0306050)

---

## License
MIT — see [LICENSE](LICENSE).
