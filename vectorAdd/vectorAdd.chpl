/*
  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
use GPU;

config const nelem: int(32) = 1024*16; //Override on the command line
var C_host: [1..nelem] real(32);
on here.gpus[0] {
  var A: [1..nelem] real(32);
  var B: [1..nelem] real(32);
  var C: [1..nelem] real(32);
  forall i in 1..nelem{
    assertOnGpu();
    A[i]=i;
  }
  forall i in 1..nelem{
    assertOnGpu();
    B[i]=2*i;
  }
  forall i in 1..nelem {
    assertOnGpu();
    C[i] = A[i] + B[i];
  }
  C_host = C;
}
var matches: bool = true;
var D: [1..nelem] real(32);
forall i in 1..nelem {
  D[i] = 3*i;
}

forall i in 1..nelem with (ref matches) {
  if (C_host[i] != D[i]) {
    writeln("Mismatch at ", i, "GPU: ", C_host[i], "CPU: ", D[i]);
    matches = false;
  }
}
if (matches) {
  writeln ("All ", nelem, " values match between CPU and GPU");
}


