HOSTNAME=$(shell hostname)
CFLAGS:=$(CFLAGS)
LDFLAGS:=$(LDFLAGS)

CHPL_CUDA_PATH=../../tool-installs/cuda-talos/
CHPL_BIN_PATH=../../tool-installs/chapel-1.28.0-build-talos/bin/

CHPL:= CHPL_CUDA_PATH=$(CHPL_CUDA_PATH) $(CHPL_BIN_PATH)/chpl
CHPL_FLAGS := $(CHPL_FLAGS) 

CUDA=nvcc
CUDA_C_FLAGS:= $(CUDA_C_FLAGS) $(CFLAGS)
CUDA_LD_FLAGS:= $(CUDA_LD_FLAGS) $(LDFLAGS)

HIPSYCL_PATH=../../tool-installs/hipSYCL-$(HOSTNAME)/install
HIPSYCL=$(HIPSYCL_PATH)/bin/syclcc
SYCL_C_FLAGS := $(SYCL_C_FLAGS) $(CFLAGS)
SYCL_LD_FLAGS := $(SYCL_LD_FLAGS) $(LDFLAGS)
HIPSYCL_C_FLAGS := $(HIPSYCL_C_FLAGS) $(SYCL_C_FLAGS) --hipsycl-targets="omp;cuda:sm_86" --cuda-path=$(CHPL_CUDA_PATH)
HIPSYCL_LD_FLAGS := $(HIPSYCL_LD_FLAGS) $(SYCL_LD_FLAGS) -Wl,-rpath=$(HIPSYCL_PATH)/lib -L$(HIPSYCL_PATH)/lib -lhipSYCL-rt -fopenmp=libomp
HIPSYCL_CHPL_LD_FLAGS := --ldflags -Wl,-rpath=$(HIPSYCL_PATH)/lib -L$(HIPSYCL_PATH)/lib -lhipSYCL-rt --ldflags -fopenmp=libomp

vecAdd: vectorAdd.chpl
	$(CHPL) vectorAdd.chpl -o vecAdd $(CHPL_FLAGS)

vecAdd-cuda: vectorAdd.chpl vecAdd-cu.o vecAdd.h
	$(CHPL) vectorAdd.chpl vecAdd.h -o vecAdd-cuda $(CHPL_FLAGS) vecAdd-cu.o

vecAdd-ocl: vectorAdd.chpl vecAdd-ocl.o vecAdd.cl
	$(CHPL) vecAdd-ocl.o $(CHPL_FLAGS)

vecAdd-hipSYCL: vecAdd-hipSYCL.o
	$(CHPL) vectorAdd.chpl vecAdd.h -o vecAdd-hipSYCL $(CHPL_FLAGS) vecAdd-hipSYCL.o $(HIPSYCL_CHPL_LD_FLAGS)

vecAdd-dpcpp: vecAdd-dpcpp.o
	$(CHPL) vecAdd-dpcpp.o vectorAdd.chpl -o vecAdd-dpcpp $(CHPL_FLAGS)

vecAdd-cu.o: vecAdd.cu
	$(CUDA) vecAdd.cu -o vecAdd-cu.o -c $(CUDA_C_FLAGS) $(CUDA_LD_FLAGS)

vecAdd-ocl.o: vecAdd.cpp vecAdd.cl
	$(CXX) vecAdd.cpp -DOPENCL -o vecAdd-ocl.o

vecAdd-hipSYCL.o: vecAdd-SYCL.cpp
	$(HIPSYCL) vecAdd-SYCL.cpp -o vecAdd-hipSYCL.o -c $(HIPSYCL_C_FLAGS)

vecAdd-dpcpp.o: vecAdd.cpp
	$(ICX) vecAdd.cpp -DSYCL -o vecAdd-dpcpp.o

clean:
	rm *.o vecAdd-dpcpp vecAdd-hipSYCL vecAdd-ocl vecAdd-cuda vecAdd
