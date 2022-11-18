/*
  Basic SYCL lambda-based implementation

  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
#ifdef __CUDACC__
  // This is only here due to an instability with the combo of LLVM-14, GNU libstdc++ 12, and CUDA
  // 11 headers
  #undef __noinline__
#endif
#include "vecAdd.h"
#include <CL/sycl.hpp>
using namespace cl;

sycl::queue myQueue;

void setQueueToSYCLDevice(int32_t dev_num) {
  // Go ahead and iterate over the SYCL devices and pick one if they've specified a device number
  std::vector<sycl::device> all_devices;
  int count = 0;
  std::vector<sycl::platform> plats = sycl::platform::get_platforms();
  for (sycl::platform plat : plats) {
    std::vector<sycl::device> devs = plat.get_devices();
    for (sycl::device dev : devs) {
      all_devices.push_back(dev);
      std::cerr << "SYCL Device [" << count << "]: " << dev.get_info<sycl::info::device::name>()
                << std::endl;
      if (count == dev_num) {
#ifdef EVENT_PROFILE
        myQueue = sycl::queue{dev, sycl::property_list{sycl::property::queue::enable_profiling()}};
#else
        myQueue = sycl::queue{dev};
#endif
      }
      ++count;
    }
  }
}

extern "C" {
void vecAdd(REAL_TYPE *A, REAL_TYPE *B, REAL_TYPE *C, int32_t lo, int32_t hi, int32_t nelem,
            int32_t dev_num) {
  int32_t work = hi-lo+1;
  setQueueToSYCLDevice(dev_num);
  // We can let the buffers use the host pointers that are coming in
  sycl::buffer<REAL_TYPE> dA(A + lo, work, sycl::property::buffer::use_host_ptr());
  sycl::buffer<REAL_TYPE> dB(B + lo, work, sycl::property::buffer::use_host_ptr());
  sycl::buffer<REAL_TYPE> dC(C + lo, work, sycl::property::buffer::use_host_ptr());
  // Since C is the result buffer, we want to force it to write back when it goes out of scope
  dC.set_write_back(true);

  // Heirarchical parallelism, but one could equally do a plain for_all without nesting
  sycl::range<1> local{256};
  sycl::range<1> global{((work / local.get(0)) + (work % local.get(0) ? 1 : 0)) * local.get(0)};
  std::cerr << "Work: " << work << " Global: " << global.get(0) << " Local: " << local.get(0)
            << std::endl;
  try {
    // Submit a lambda which 1) gets access to the buffers, and 2) invokes the kernel
    sycl::event vecAdd_event = myQueue.submit([&](sycl::handler &cgh) {
      // Get access to data ranges
      sycl::accessor<REAL_TYPE, 1, sycl::access::mode::read> A_acc =
          dA.get_access<sycl::access::mode::read>(cgh, sycl::range<1>{(size_t)work});
      sycl::accessor<REAL_TYPE, 1, sycl::access::mode::read> B_acc =
          dB.get_access<sycl::access::mode::read>(cgh, sycl::range<1>{(size_t)work});
      sycl::accessor<REAL_TYPE, 1, sycl::access_mode::discard_write> C_acc =
          dC.get_access<sycl::access::mode::discard_write>(cgh,
                                                           sycl::range<1>{(size_t)work});
      // Invoke the kernel, which is also a lambda here (but could be a functor
      // Needs to be by-value, since we'll transition to the device address space
      cgh.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> tid_info) {
        // This is the kernel body
        size_t tid = tid_info.get_global_linear_id();
        if (tid < work) {
          // Use the accessors
          C_acc[tid] = A_acc[tid] + B_acc[tid];
        }
      });
    });
#ifdef EVENT_PROFILE
    vecAdd_event.wait();
    auto end = vecAdd_event.get_profiling_info<sycl::info::event_profiling::command_end>();
    auto start = vecAdd_event.get_profiling_info<sycl::info::event_profiling::command_start>();
    std::cerr << "VecAdd kernel elapsed time: " << (end - start) << " (ns)" << std::endl;
#endif
    myQueue.wait();
  } catch (sycl::exception e) {
    std::cerr << "SYCL Exception during Kernel\n\t" << e.what() << std::endl;
  }
}
}
