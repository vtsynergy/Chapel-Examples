HOSTNAME=$(shell hostname)
CFLAGS:=$(CFLAGS)
LDFLAGS:=$(LDFLAGS)
CC=clang
CXX=clang++

CHPL_CUDA_PATH=../../tool-installs/cuda-$(HOSTNAME)
CHPL_BIN_PATH=../../tool-installs/chapel-latest-$(HOSTNAME)/bin

CHPL:= CHPL_CUDA_PATH=$(CHPL_CUDA_PATH) $(CHPL_BIN_PATH)/chpl
CHPL_FLAGS := $(CHPL_FLAGS) $(CHPL_GPUAPI_MODULES) 

#CHPL_FLAGS := $(CHPL_FLAGS) -g  --no-optimize --baseline
CHPL_FLAGS := $(CHPL_FLAGS) -g --devel


.PHONY: all
all: jaccard

jaccard: jaccard.chpl CSR.m EdgeCentric.m CuGraph.m
	$(CHPL) jaccard.chpl -o jaccard $(CHPL_FLAGS)

CSR.m: CSR.chpl
	$(CHPL) CSR.chpl -o CSR.m $(CHPL_FLAGS)

EdgeCentric.m: EdgeCentric.chpl
	$(CHPL) EdgeCentric.chpl -o EdgeCentric.m $(CHPL_FLAGS)

CuGraph.m: CuGraph.chpl
	$(CHPL) CuGraph.chpl -o CuGraph.m $(CHPL_FLAGS)

.PHONY: clean
clean:
	rm *.m jaccard
