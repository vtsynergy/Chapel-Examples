/*
  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
module vecAdd {
  use GPU;

  proc vecAdd(A_host: [] real(32), B_host: [] real(32), C_host: [] real(32), lo: int(32), hi: int(32), nelem: int(32), dev_num: int(32)) {

    var foo : real(32) = A_host(1);

    on here.gpus[dev_num] {
      var A: [lo+1..hi+1] real(32);
      var B: [lo+1..hi+1] real(32);
      var C: [lo+1..hi+1] real(32);
      A = A_host;
      B = B_host;
      forall i in lo+1..hi+1 {
        assertOnGpu();
        C[i] = A[i] + B[i];
      }
      C_host = C;
    }
  }
}
