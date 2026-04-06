# Makefile for LLM C Transformer with TCO Benchmarking
# 
# Targets:
#   all         - Build all binaries
#   lm          - Language model training
#   train       - NER training
#   bench       - Inference benchmark
#   tco_bench   - TCO benchmark suite (NEW)
#   clean       - Remove build artifacts

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -lm -lgomp

# Source directories
SRC_DIR = src
BUILD_DIR = build
RESULTS_DIR = results

# Main targets
TARGETS = lm train bench tco_bench

.PHONY: all clean run_tco_analysis

all: $(TARGETS)

# Language Model training
lm: $(SRC_DIR)/lm_train.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# NER training
train: $(SRC_DIR)/main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Inference benchmark
bench: $(SRC_DIR)/bench.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# TCO benchmark (NEW)
tco_bench: tco_bench.c
	@echo "Building TCO benchmark..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "TCO benchmark built successfully!"

# Run complete TCO analysis
run_tco_analysis: tco_bench
	@echo "Creating results directories..."
	@mkdir -p $(RESULTS_DIR)/tco
	@echo ""
	@echo "Running TCO benchmarks..."
	./tco_bench
	@echo ""
	@echo "Generating TCO analysis and plots..."
	python3 scripts/tco_analysis.py --scenarios all
	@echo ""
	@echo "TCO analysis complete! Check $(RESULTS_DIR)/tco/ for results."

# Clean build artifacts
clean:
	rm -f $(TARGETS)
	rm -rf $(BUILD_DIR)
	@echo "Clean complete."

# Help
help:
	@echo "LLM C Transformer Build System"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all binaries"
	@echo "  lm               - Build language model training binary"
	@echo "  train            - Build NER training binary"
	@echo "  bench            - Build inference benchmark"
	@echo "  tco_bench        - Build TCO benchmark suite (NEW)"
	@echo "  run_tco_analysis - Run complete TCO benchmark + analysis (NEW)"
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Example workflows:"
	@echo "  make all && ./lm              # Train language model"
	@echo "  make bench && ./bench         # Run inference benchmark"
	@echo "  make run_tco_analysis         # Complete TCO analysis"
