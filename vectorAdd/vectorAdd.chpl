/*
  This is the main entrypoint and CPU validation
  Accel. Impls. including Chapel's built-in GPU support are externally linked

  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/
use GPU;

extern proc vecAdd(A: [] real(32), B: [] real(32), C: [] real(32), lo: int(32), hi: int(32), nelem: int(32));

config const nelem: int(32) = 1024*16; //Override on the command line
var A_host: [1..nelem] real(32);
var B_host: [1..nelem] real(32);
var C_host: [1..nelem] real(32);

//Initialize data on the GPU for convenience
on here.gpus[0] {
  var A: [1..nelem] real(32);
  var B: [1..nelem] real(32);
  forall i in 1..nelem{
    assertOnGpu();
    A[i]=i;
  }
  A_host = A;
  forall i in 1..nelem{
    assertOnGpu();
    B[i]=2*i;
  }
  B_host = B;
}

//Not sure how to use the iterator yet, lets start with a direct call
//Whose job is it to convert from base-0 to base-1 indexing and back?
//Assuming we are modeling calling existing backend functions, then we should convert in Chapel space, not expect the library code to convert.
vecAdd(A_host, B_host, C_host, 0, nelem-1, nelem);


var matches: bool = true;
var D: [1..nelem] real(32);
forall i in 1..nelem {
  D[i] = 3*i;
}

forall i in 1..nelem with (ref matches) {
  if (C_host[i] != D[i]) {
    writeln("Mismatch at ", i, " GPU: ", C_host[i], " CPU: ", D[i]);
    matches = false;
  }
}
if (matches) {
  writeln ("All ", nelem, " values match between CPU and GPU");
}
writeln(C_host[1..10]);
