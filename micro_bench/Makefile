ARCH ?= sm_90
NVCC ?= nvcc
FLAGS = -O3 -arch=$(ARCH)
SRCS = $(wildcard m_*.cu)
BINS = $(SRCS:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) -o $@ $<

clean:
	rm -f $(BINS)

.PHONY: all clean
