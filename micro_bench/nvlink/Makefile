ARCH  ?= sm_90
NVCC  ?= nvcc
FLAGS  = -O3 -arch=$(ARCH) -lineinfo
LIBS   = -lnvidia-ml -lcuda

SRCS = $(wildcard nv*.cu)
BINS = $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin:
	@mkdir -p bin

bin/%: %.cu nvl_common.cuh | bin
	$(NVCC) $(FLAGS) -o $@ $< $(LIBS)

# 串行跑全部实验(GPU 独占), 日志落到 raw_logs/
run: all
	@mkdir -p raw_logs
	@for b in $(BINS); do \
	  n=`basename $$b`; \
	  echo "=== $$n ==="; \
	  ./runlock.sh $$b > raw_logs/$$n.log 2>&1 || echo "  FAILED"; \
	  tail -3 raw_logs/$$n.log; \
	done

clean:
	rm -rf bin

.PHONY: all clean run
