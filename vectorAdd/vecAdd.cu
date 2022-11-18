#include "vecAdd.h"

__global__ void vectorAddCUDA(REAL_TYPE *A, REAL_TYPE *B, REAL_TYPE *C, int32_t nelem) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < nelem) {
    C[tid] = A[tid] + B[tid];
  }
}

extern "C" {
void vecAdd(REAL_TYPE *A, REAL_TYPE *B, REAL_TYPE *C, int32_t lo, int32_t hi, int32_t nelem) {
  REAL_TYPE *dA, *dB, *dC;
  int32_t work = hi-lo+1;
  cudaSetDevice(dev_num);
  cudaMalloc(&dA, sizeof(REAL_TYPE) * work);
  cudaMalloc(&dB, sizeof(REAL_TYPE) * work);
  cudaMalloc(&dC, sizeof(REAL_TYPE) * work);

  cudaMemcpy(dA, A + lo, sizeof(REAL_TYPE) * work, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B + lo, sizeof(REAL_TYPE) * work, cudaMemcpyHostToDevice);
  dim3 block = {256, 1, 1};
  dim3 grid = {(work / block.x) + (work % block.x ? 1 : 0), 1, 1};
  vectorAddCUDA<<<grid, block>>>(dA, dB, dC, work);
  cudaMemcpy(C + lo, dC, sizeof(REAL_TYPE) * work, cudaMemcpyDeviceToHost);
}
}
