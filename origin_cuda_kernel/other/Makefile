NVCC ?= nvcc
ARCH ?= sm_120
FLAGS := -O3 -std=c++17 -arch=$(ARCH)

.PHONY: all scan sort topk clean

all: scan sort topk

scan: scan/scan
scan/scan: scan/scan.cu
	$(NVCC) $(FLAGS) -o $@ $<

sort: sorting/sort
sorting/sort: sorting/sorting.cu
	$(NVCC) $(FLAGS) -o $@ $<

topk: topk/topk
topk/topk: topk/topk.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f scan/scan sorting/sort topk/topk
