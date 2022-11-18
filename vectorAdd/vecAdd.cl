/*
  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
#ifndef REAL_TYPE
#define REAL_TYPE float
#endif

__kernel void vectorAddOpenCL(__global REAL_TYPE *A, __global REAL_TYPE *B, __global REAL_TYPE *C, int nelem) {
  size_t tid = get_global_id(0);
  if (tid < nelem) {
    C[tid] = A[tid] + B[tid];
  }
}
