CHPL_CUDA_PATH=../../tool-installs/cuda-talos/
CHPL_BIN_PATH=../../tool-installs/chapel-1.28.0-build-talos/bin/

CHPL=$(CHPL_BIN_PATH)/chpl
CHPL_FLAGS := $(CHPL_FLAGS) 


vecAdd: vectorAdd.chpl
	$(CHPL) vectorAdd.chpl -o vecAdd $(CHPL_FLAGS)

vecAdd-cuda: vectorAdd.chpl vecAdd-cu.o
	$(CHPL) vecAdd-cu.o vectorAdd.chpl -o vecAdd-cuda $(CHPL_FLAGS)

vecAdd-ocl: vectorAdd.chpl vecAdd-ocl.o vecAdd.cl
	$(CHPL) vecAdd-ocl.o $(CHPL_FLAGS)

vecAdd-hipSYCL: vecAdd-hipSYCL.o
	$(CHPL) vecAdd-hipSYCL.o vectorAdd.chpl -o vecAdd-hipSYCL $(CHPL_FLAGS)

vecAdd-dpcpp: vecAdd-dpcpp.o
	$(CHPL) vecAdd-dpcpp.o vectorAdd.chpl -o vecAdd-dpcpp $(CHPL_FLAGS)

vecAdd-cu.o: vecAdd.cu
	$(CUDA) vecAdd.cu -o vecAdd-cu.o

vecAdd-ocl.o: vecAdd.cpp vecAdd.cl
	$(CXX) vecAdd.cpp -DOPENCL -o vecAdd-ocl.o

vecAdd-hipSYCL.o: vecAdd.cpp
	$(HIPSYCL) vecAdd.cpp -DSYCL -o vecAdd-hipSYCL.o

vecAdd-dpcpp.o: vecAdd.cpp
	$(ICX) vecAdd.cpp -DSYCL -o vecAdd-dpcpp.o


