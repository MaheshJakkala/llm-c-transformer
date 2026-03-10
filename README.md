# Transformer-Based LLM from Scratch in C

<div align="center">

![Language](https://img.shields.io/badge/Language-C11-blue?logo=c)
![SIMD](https://img.shields.io/badge/SIMD-AVX2%2FFMA-orange)
![OpenMP](https://img.shields.io/badge/Parallel-OpenMP-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Designed and implemented a transformer-based LLM from scratch in C — custom tensor library, INT8 post-training quantization with AVX2 SIMD, memory-efficient training pipeline (tensor arena allocator), and benchmarked performance improvements over PyTorch FP32 and PyTorch INT8 baselines.**

</div>

---

## What This Project Is

This project implements the **full transformer stack in pure C** — no ML framework, no BLAS, no external dependencies beyond libc.

| Component | File | Task |
|-----------|------|------|
| **Causal Language Model** | `src/lm_train.c` | Next-token prediction on raw text. Given 32 chars, predict the 33rd. This is the LLM component. |
| **NER Fine-Tuning** | `src/main.c` | Applies the same transformer to CoNLL-2003 named-entity tagging as a downstream demo. |
| **Inference Benchmark** | `src/bench.c` | C FP32 vs C INT8-AVX2 — real measurements, CSV output. |
| **Honest Speedup Proof** | `benchmark_pytorch.py` | PyTorch FP32 vs PyTorch INT8 vs C FP32 vs C INT8-AVX2 — all 4 backends, same model, same hardware. |

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

Total training: **20.04 s** · **~800 ms/epoch** · **11,200 malloc calls eliminated per epoch** (arena)

---

### Honest Inference Benchmark: All 4 Backends

> All numbers measured on the same CPU, same model (HIDDEN=256, FFN=512), same protocol  
> (10 warm-up + 100 timed iterations, single-threaded, averaged).  
> Reproducible: run `./run_benchmark.sh` to regenerate every number.

| Seq | PyTorch FP32 (ms) | PyTorch INT8 (ms) | C FP32 (ms) | C INT8-AVX2 (ms) | vs PT FP32 | vs PT INT8 |
|-----|:-----------------:|:-----------------:|:-----------:|:----------------:|:----------:|:----------:|
| 8   | 2.014             | 2.077             | 4.686       | **0.128**        | **15.8×**  | **16.3×**  |
| 16  | 3.157             | 3.725             | 9.287       | **0.240**        | **13.2×**  | **15.5×**  |
| 32  | 5.048             | 5.007             | 19.075      | **0.627**        | **8.1×**   | **8.0×**   |
| 64  | 8.258             | 8.917             | 38.722      | **3.725**        | **2.2×**   | **2.4×**   |

**Why PyTorch INT8 ≈ PyTorch FP32:** `torch.quantization.quantize_dynamic` stores weights as INT8 but immediately dequantizes them back to FP32 before every matmul. The compute is still FP32. This is why the "optimal" PyTorch INT8 baseline gives ~0% speedup over plain FP32.

**Why C INT8-AVX2 wins against both:** weights stay INT8 through accumulation. `_mm256_maddubs_epi16` executes 32 INT8 multiply-accumulates per SIMD instruction. 4× smaller weights fit entirely in L1/L2 cache, eliminating cache misses that FP32 suffers.

> The advantage shrinks at seq=64 because attention scores (seq × seq) are not quantized and scale quadratically.

---

### MatMul Micro-Benchmark (M=N=K=256)

| Backend           | Time (ms) | Throughput        |
|-------------------|:---------:|:-----------------:|
| NumPy/PyTorch CPU | 0.376 ms  | 89.33 GFLOP/s     |
| C FP32 Naive      | 2.772 ms  | 12.11 GFLOP/s     |
| **C INT8-AVX2**   | **0.572 ms** | **58.68 GOPS/s** |

INT8-AVX2 is **4.85× faster than C FP32** and **0.66× of NumPy** at the isolated matmul level.  
Note: NumPy/PyTorch use OpenBLAS (highly tuned BLAS), so the full-pipeline speedup comes from the combined effect of SIMD quantization + cache locality + zero framework overhead.

---

### Memory Footprint

| Precision | Transformer Weights | Reduction |
|-----------|:-------------------:|:---------:|
| FP32      | 2.01 MB             | baseline  |
| **INT8**  | **0.50 MB**         | **4.00×** |

---

### Memory Arena — Training Allocator

| Method   | malloc/free per step | Behaviour |
|----------|:--------------------:|-----------|
| Standard | 512 calls            | heap fragmentation, cold cache |
| **Arena**| **0 calls** (O(1) pool reset) | **11,200 calls eliminated/epoch** · 0.14 MB peak / 0.75 MB reserved |

---

## Plots

<table>
<tr>
<td align="center"><img src="results/plots/full_comparison.png"/><br><b>★ Honest 4-Backend Comparison</b></td>
</tr>
<tr>
<td align="center"><em>PyTorch FP32 | PyTorch INT8 | C FP32 | C INT8-AVX2 — same model, same hardware, reproducible</em></td>
</tr>
</table>

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
│   ├── bench.c             ←   C FP32 vs C INT8-AVX2 benchmark binary
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
│   ├── plots/              ←   8 benchmark plots (real measurements)
│   └── metrics/            ←   CSV files + training/benchmark logs
├── notebooks/
│   └── LLMFromScratch.ipynb  ← PyTorch reference implementation + baseline
├── scripts/
│   └── plot_results.py     ←   Regenerates all 8 plots from CSV data
├── benchmark_pytorch.py    ← ★ Honest 4-backend benchmark script
|__ BENCHMARK.md            <-- clear details regarding benchmarking against PyTorch
├── run_benchmark.sh        ← ★ One command: build → bench → compare → plot
└── Makefile
```

---

## Build & Run

```bash
# Ubuntu / Debian — GCC ≥ 10 required
sudo apt install gcc libgomp1 make python3 python3-pip

# One command: builds C, runs C benchmark, runs Python benchmark, regenerates all plots
./run_benchmark.sh

# Or step by step:
make all                        # builds: lm, train, bench
./lm                            # train the language model
./bench                         # C FP32 vs C INT8-AVX2
./train                         # NER fine-tuning demo
python3 benchmark_pytorch.py    # PyTorch FP32 + INT8 vs C INT8-AVX2
python3 scripts/plot_results.py # regenerate all 8 plots
```

### Requirements for full benchmark

| Tool | Purpose |
|------|---------|
| GCC ≥ 10 + AVX2 CPU | Build C binaries with SIMD |
| `python3 torch` | PyTorch FP32 + INT8 baseline (preferred) |
| `python3 numpy` | NumPy fallback if PyTorch unavailable (same OpenBLAS backend) |

`benchmark_pytorch.py` auto-detects PyTorch and falls back to NumPy if not installed. NumPy uses OpenBLAS, the same BLAS backend PyTorch CPU calls for `nn.Linear`, so results are equivalent.

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
Eliminates **11,200 malloc/free calls per epoch**. Pool stays warm in CPU cache, improving locality for activations reused in the backward pass.

### INT8 Quantization + AVX2 MatMul (`src/qtensor.c`)
```c
// Symmetric per-tensor quantization: scale = max_abs / 127
QTensor *quantize_weight_transpose(const Tensor *W);

// 32 INT8 multiply-accumulates per SIMD instruction
void matmul_q8_avx2(const int8_t *A, const int8_t *B, int32_t *C,
                    int M, int N, int K);
// _mm256_maddubs_epi16 + _mm256_madd_epi16
// 4.85× faster than scalar C FP32, 4.00× less memory
```

### Full Backpropagation
```
cross_entropy_grad
  → linear_backward (W_lm)
    → ffn_backward (W2, W1)
      → attention_backward (Wo, Wv, Wk, Wq)
        → embedding_backward
```
All gradients hand-derived. Adam with L2 gradient clipping applied after each step.

---

## vs PyTorch Baseline — Proven by the Repository

`benchmark_pytorch.py` runs all four backends in one script and proves the claim:

| Metric | PyTorch FP32 | PyTorch INT8 | C INT8-AVX2 |
|--------|:------------:|:------------:|:-----------:|
| Weight memory | 2.01 MB | 2.01 MB* | **0.50 MB (4×)** |
| Latency (seq=16) | 3.157 ms | 3.725 ms | **0.240 ms (13×)** |
| Training allocator | malloc/free per step | malloc/free per step | **Arena pool (O(1) reset)** |
| Dependencies | PyTorch, NumPy, BLAS | PyTorch, NumPy, BLAS | libc, libm, libgomp only |

\* `quantize_dynamic` reduces storage but dequantizes weights to FP32 before every matmul, so runtime memory and speed are the same as FP32.

**The speedup is real and holds against the optimal PyTorch INT8 baseline.** To reproduce: `./run_benchmark.sh`

---

## References

- Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Dettmers et al. (2022). [LLM.int8()](https://arxiv.org/abs/2208.07339)
- Karpathy, A. [llm.c](https://github.com/karpathy/llm.c) — inspiration for this project
- Sang & De Meulder (2003). [CoNLL-2003 NER Shared Task](https://arxiv.org/abs/cs/0306050)

---

## License
MIT — see [LICENSE](LICENSE).
