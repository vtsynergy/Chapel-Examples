/*
  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
use GPU;

//This is "library" code which we are assuming uses C conventions (base-0 indexing)
proc vecAdd(A_host: [] real(32), B_host: [] real(32), C_host: [] real(32), lo: int(32), hi: int(32), nelem: int(32)) {

  on here.gpus[0] {
    var A: [lo..hi] real(32);
    var B: [lo..hi] real(32);
    var C: [lo..hi] real(32);
    forall i in lo..hi {
      assertOnGpu();
      C[i] = A[i] + B[i];
    }
    C_host = C;
  }
}
