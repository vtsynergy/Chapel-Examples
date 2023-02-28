module Jaccard {
  use CSR;
  use EdgeCentric;
  use CuGraph;

  //TODO Map the "ENABLE" and "DISABLE" preprocessor behavior from C++ to `config param`

  //Need to expose device selection to the command line

  //Need to expose pipeline selection to the command line

  //Need to write a separate bridge module to the SYCL implementations (far later)

  //Should probably wrap the kernel pipeline prototypes in separate modules

  config var inFile = "" : string;
  config var outFile  = "" : string;
  config const devNum = 0 : int;
  config const useCUGraph = false : bool;
  config const useWeighted = false : bool;

  proc main() {
  //Make sure we have some data to process and somewhere to put it
assert(inFile != "", "Must provide input file with '--inFile=<pathToFile>'");
assert(outFile != "", "Must provide output file with '--outFile=<pathToFile>'");

  //Read the input data into a CSR member
  //readCSRFile(inFile);
  CSRUser(inFile);
//  var inCSR = readCSRFile(inFile);
    //Read the input file and set up host arrays (use generic methods to support different FP types)
    //Launch the selected kernel pipeline
    //Write the output file
  if (useCUGraph) {
    //do VC stuff
  } else {
    //Do EC stuff
    var myBlandCSR : unmanaged CSR(false, false, false, false)?;
    var myBlandWeightsDom : domain(1) = {1..10};
    var myBlandWeights : [myBlandWeightsDom] real(32);
//    EC_Jaccard(CSR(false, false, false, false), myBlandCSR, real(32), myBlandWeights); 
    var myBlandCSR2 : unmanaged CSR(true, true, true, true)?;
    var myBlandWeightsDom2 : domain(1) = {1..10};
    var myBlandWeights2 : [myBlandWeightsDom2] real(32);
//    EC_Jaccard(CSR(true, true, true, true), myBlandCSR2, real(32), myBlandWeights2); 
    var myBlandCSR3 : unmanaged CSR(isVertexT64 = true, isEdgeT64=true, false, false)?;
    var myBlandWeightsDom3 : domain(1) = {1..10};
    var myBlandWeights3 : [myBlandWeightsDom3] real(32);
//    EC_Jaccard(CSR(isVertexT64 = true, isEdgeT64=true, false, false), myBlandCSR3, real(32), myBlandWeights3); 
  }
  }
}
