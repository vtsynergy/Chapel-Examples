/*
  This file will eventually include both the OpenCL and SYCL host implementations

  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
#include <cstdint>
#include "vecAdd.h"
#include "metamorph.h"
#include <CL/opencl.h>
#include "metamorph_opencl.h"
#include "metacl_vecAdd.h"
#define STRINGIZE(A) #A
#define STR(A) STRINGIZE(A)

extern "C" {
void vecAdd(REAL_TYPE *A, REAL_TYPE *B, REAL_TYPE *C, int32_t lo, int32_t hi, int32_t nelem, int32_t dev_num) {
  int32_t work = hi-lo+1;
  //Set up device
  meta_set_acc(dev_num, metaModePreferOpenCL);
  //Add build args
  __metacl_vecAdd_custom_args = "-D REAL_TYPE=" STR(REAL_TYPE) ;
  //Get the queue and context
  cl_device_id dev;
  cl_platform_id plat;
  cl_context ctx;
  cl_command_queue q;
  meta_get_state_OpenCL(&plat, &dev, &ctx, &q);
  //Create buffers
  cl_mem dA, dB, dC;
  dA = clCreateBuffer(ctx, NULL, sizeof(REAL_TYPE) * work, NULL, NULL);
  dB = clCreateBuffer(ctx, NULL, sizeof(REAL_TYPE) * work, NULL, NULL);
  dC = clCreateBuffer(ctx, NULL, sizeof(REAL_TYPE) * work, NULL, NULL);
  //Copy buffers
  clEnqueueWriteBuffer(q, dA, CL_FALSE, 0, sizeof(REAL_TYPE) * work, A, 0, NULL, NULL);
  clEnqueueWriteBuffer(q, dB, CL_TRUE, 0, sizeof(REAL_TYPE) * work, B, 0, NULL, NULL);
  //Invoke
  size_t local[3] = {256, 1, 1};
  size_t global[3] = {((work / local[0]) + (work % local[0] ? 1 : 0)) * local[0], 1, 1};
  metacl_vecAdd_vectorAddOpenCL(q, &global, &local, NULL, false, NULL, &dA, &dB, &dC, work);
  //Copy buffers
  clEnqueueReadBuffer(q, dC, CL_TRUE, 0, sizeof(REAL_TYPE) * work, C, 0, NULL, NULL);
  clFinish(q);
  //Release buffers
  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
}
}
