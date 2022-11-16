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
  cudaMalloc(&dA, sizeof(REAL_TYPE) * (hi - lo + 1));
  cudaMalloc(&dB, sizeof(REAL_TYPE) * (hi - lo + 1));
  cudaMalloc(&dC, sizeof(REAL_TYPE) * (hi - lo + 1));

  cudaMemcpy(dA, A + lo, sizeof(REAL_TYPE) * (hi - lo + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B + lo, sizeof(REAL_TYPE) * (hi - lo + 1), cudaMemcpyHostToDevice);
  dim3 block = {256, 1, 1};
  dim3 grid = {(nelem / block.x) + (nelem % block.x ? 1 : 0), 1, 1};
  vectorAddCUDA<<<grid, block>>>(dA, dB, dC, nelem);
  cudaMemcpy(C + lo, dC, sizeof(REAL_TYPE) * (hi - lo + 1), cudaMemcpyDeviceToHost);
}
}
