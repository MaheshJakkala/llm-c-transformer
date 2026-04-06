# TCO Quick Reference Card

## 🎯 One-Page Summary: When to Use What

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT DECISION TREE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Volume < 100M tokens/month? ──────────────────► C INT8-AVX2 (CPU)     │
│                                                   Cheapest: $872/year   │
│                                                                         │
│  Edge/IoT/Mobile? ─────────────────────────────► C INT8-AVX2 (CPU)     │
│                                                   Only viable option    │
│                                                                         │
│  AWS Lambda / Serverless? ─────────────────────► C INT8-AVX2 (CPU)     │
│                                                   Fast cold start       │
│                                                                         │
│  Volume 100M - 10B tokens/month? ──────────────► Compare both:         │
│    • C INT8-AVX2: Better for variable load      $0.011/1M tokens       │
│    • GPU T4: Better for batch processing        $0.026/1M tokens       │
│                                                                         │
│  Volume > 10B tokens/month? ────────────────────► GPU (T4/A100)        │
│                                                   Scale amortizes cost  │
│                                                                         │
│  Need < 100μs latency? ─────────────────────────► GPU (A100)           │
│                                                   Only option           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Comparison at a Glance

| Metric | C INT8-AVX2 | PyTorch CPU | GPU T4 | GPU A100 |
|--------|-------------|-------------|--------|----------|
| **Latency** (seq=16) | 0.275 ms | 2.355 ms | ~0.05 ms | ~0.02 ms |
| **Throughput** | 3,636 tok/s | 425 tok/s | ~20,000 tok/s | ~50,000 tok/s |
| **Memory** | 0.50 MB | 2.01 MB | 2.01 MB | 2.01 MB |
| **$/1M tokens** | **$0.011** 🏆 | $0.098 | $0.026 | $0.061 |
| **Perf/$** | **18.2** 🏆 | 2.1 | 10.0 | 5.0 |
| **TCO/year** (1B tok/mo) | **$872** 🏆 | $2,189 | $2,538 | $7,344 |
| **Best For** | Edge, Serverless | Development | Cloud Batch | Enterprise |

🏆 = Winner in that category

---

## 💰 Cost Breakdown (1B tokens/month, 1 year)

```
C INT8-AVX2:  ████░░░░░░░░░░░░░░░░  $872   (baseline)
PyTorch CPU:  ███████████░░░░░░░░░  $2,189 (2.5× more)
GPU T4:       ████████████░░░░░░░░  $2,538 (2.9× more)
GPU A100:     ███████████████████   $7,344 (8.4× more)
```

---

## ⚡ Speed Comparison

```
LATENCY (Lower is Better):
GPU A100:     ████░░░░░░░░░░░░░░░░  0.02 ms   (fastest)
GPU T4:       ██████░░░░░░░░░░░░░░  0.05 ms
C INT8-AVX2:  ███████████░░░░░░░░░  0.275 ms
PyTorch CPU:  ████████████████████  2.355 ms  (slowest)

THROUGHPUT (Higher is Better):
GPU A100:     ████████████████████  50,000 tok/s  (fastest)
GPU T4:       ████████░░░░░░░░░░░░  20,000 tok/s
C INT8-AVX2:  ███░░░░░░░░░░░░░░░░░   3,636 tok/s
PyTorch CPU:  ░░░░░░░░░░░░░░░░░░░░     425 tok/s  (slowest)
```

---

## 🎯 Key Insights

### C INT8-AVX2 Wins On:
- ✅ **Cost per token** (8.6× cheaper than PyTorch)
- ✅ **Memory efficiency** (4× smaller model)
- ✅ **Performance per dollar** (8.7× better than GPU)
- ✅ **Edge deployment** (no GPU alternative)
- ✅ **Serverless** (fast cold start, low memory)

### GPU Wins On:
- ✅ **Raw throughput** (5.5× faster than C INT8-AVX2)
- ✅ **Ultra-low latency** (< 100 μs for A100)
- ✅ **High volume** (> 10B tokens/month amortizes cost)
- ✅ **Batch processing** (parallel inference)

### PyTorch CPU:
- ⚠️ **Rarely optimal** for production
- ✅ Good for: Development, debugging, prototyping
- ❌ Expensive for production inference

---

## 🔢 ROI Examples

### Scenario 1: Startup Chatbot (100M tokens/month)

| System | Monthly Cost | Annual Cost | Savings vs PyTorch |
|--------|--------------|-------------|-------------------|
| **C INT8-AVX2** | **$73** | **$876** | Baseline |
| PyTorch CPU | $183 | $2,196 | -$1,320/year 💸 |
| GPU T4 | $212 | $2,544 | -$1,668/year 💸 |

**Winner**: C INT8-AVX2 saves $1,320-1,668/year

---

### Scenario 2: Enterprise API (10B tokens/month)

| System | Monthly Cost | Annual Cost | Savings vs PyTorch |
|--------|--------------|-------------|-------------------|
| C INT8-AVX2 | $1,144 | $13,728 | Baseline |
| PyTorch CPU | $9,792 | $117,504 | -$103,776/year 💸💸💸 |
| **GPU T4** | **$2,628** | **$31,536** | +$82,968/year ✅ |

**Winner**: GPU T4 at this scale (but C INT8 still 2.3× cheaper)

---

### Scenario 3: Edge IoT Device (10M tokens/month)

| System | Monthly Cost | Annual Cost | Notes |
|--------|--------------|-------------|-------|
| **C INT8-AVX2** | **$51** | **$612** | Only option |
| PyTorch CPU | N/A | N/A | Too large for edge |
| GPU T4 | N/A | N/A | No GPU on IoT devices |

**Winner**: C INT8-AVX2 (no alternative exists)

---

## 🚀 Quick Start Commands

```bash
# Install and run complete TCO analysis
git clone https://github.com/MaheshJakkala/llm-c-transformer
cd llm-c-transformer

# Build everything
make all

# Run TCO benchmark + analysis
make run_tco_analysis

# View results
ls results/tco/
open results/tco/tco_summary_all_scenarios.png
```

---

## 📋 Checklist: Before Deploying to Production

**For C INT8-AVX2:**
- [ ] Measured latency on target hardware
- [ ] Validated accuracy vs FP32 baseline
- [ ] Confirmed CPU supports AVX2 (check: `lscpu | grep avx2`)
- [ ] Calculated TCO for your expected volume
- [ ] Tested cold start time for serverless

**For GPU Deployment:**
- [ ] Measured actual GPU latency (not estimates)
- [ ] Calculated cost at 80%+ utilization
- [ ] Confirmed batch size for efficiency
- [ ] Validated power/cooling requirements
- [ ] Budgeted for CUDA optimization time

**For PyTorch CPU:**
- [ ] Confirmed it's only for development
- [ ] Have migration plan to C or GPU for production
- [ ] Understand 8.6× cost premium

---

## 🔗 Quick Links

- **Full Documentation**: `TCO_GUIDE.md`
- **Implementation Details**: `README.md`
- **Source Code**: `src/` directory
- **TCO Script**: `scripts/tco_analysis.py`
- **Benchmark Code**: `tco_bench.c`

---

## 📞 Getting Help

**Question**: "Which system should I use for my use case?"
**Answer**: Run `make run_tco_analysis` and check `results/tco/` outputs.

**Question**: "How do I update with my own measurements?"
**Answer**: Edit `scripts/tco_analysis.py` → `SYSTEMS` config.

**Question**: "GPU estimates don't match my hardware"
**Answer**: Measure on your GPU, update config, re-run analysis.

**Question**: "Can I use this for other models?"
**Answer**: Yes! Update latency/throughput in config.

---

**Last Updated**: April 2026  
**Print this page for quick reference!**
