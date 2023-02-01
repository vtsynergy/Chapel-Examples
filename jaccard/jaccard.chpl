module Jaccard {
  use CSR;
  use EdgeCentric;
  use CuGraph;

  //Need to expose device selection to the command line

  //Need to expose pipeline selection to the command line

  //Need to write a separate bridge module to the SYCL implementations (far later)

  //Should probably wrap the kernel pipeline prototypes in separate modules

  config var inFile : string;
  config var outFile : string;
  config const devNum = 0 : int;
  config const useCUGraph = false : bool;

  proc main() {
    //Read the input file and set up host arrays (use generic methods to support different FP types)
    //Launch the selected kernel pipeline
    //Write the output file
  }
}
