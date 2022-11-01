/*
  Copyright 2022 Virginia Tech
  Author: Paul Sathre
*/

config const nelem: int(32) = 1024*16; //Override on the command line
var A: [1..nelem] real(32);
var B: [1..nelem] real(32);
var C: [1..nelem] real(32);
forall i in 1..nelem{
  A[i]=i;
}
forall i in 1..nelem{
  B[i]=2*i;
}
forall i in 1..nelem {
  C[i] = A[i] + B[i];
}
