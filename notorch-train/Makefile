# Janus Sonar — notorch-based training pipeline
# "fuck torch"

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
  BLAS_NAME = Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
  BLAS_NAME = OpenBLAS
endif

.PHONY: all clean

all: train_janus_sonar infer_janus_sonar infer_janus_sonar_chain train_janus_sft

train_janus_sonar: train_janus_sonar.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_janus_sonar train_janus_sonar.c notorch.c -lm
	@echo "Compiled: train_janus_sonar (dual weights, $(BLAS_NAME))"

infer_janus_sonar: infer_janus_sonar.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_sonar infer_janus_sonar.c notorch.c -lm
	@echo "Compiled: infer_janus_sonar (dual weights, $(BLAS_NAME))"

infer_janus_sonar_chain: infer_janus_sonar_chain.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_sonar_chain infer_janus_sonar_chain.c notorch.c -lm
	@echo "Compiled: infer_janus_sonar_chain (8-step bidir + SPA, $(BLAS_NAME))"

train_janus_sft: train_janus_sft.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o train_janus_sft train_janus_sft.c notorch.c -lm
	@echo "Compiled: train_janus_sft (LoRA rank-8 Q/K/V, $(BLAS_NAME))"

infer_janus_sft: infer_janus_sft.c notorch.c notorch.h
	$(CC) $(CFLAGS) $(BLAS_FLAGS) -o infer_janus_sft infer_janus_sft.c notorch.c -lm
	@echo "Compiled: infer_janus_sft (base + LoRA adapter, $(BLAS_NAME))"

clean:
	rm -f train_janus_sonar infer_janus_sonar *.o
