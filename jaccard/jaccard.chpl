module Jaccard {
  use CSR;
  use EdgeCentric;
  use CuGraph;

  //Need to expose device selection to the command line

  //Need to expose pipeline selection to the command line

  //Need to write a separate bridge module to the SYCL implementations (far later)

  //Should probably wrap the kernel pipeline prototypes in separate modules

  config var inFile = "" : string;
  config var outFile  = "" : string;
  config const devNum = 0 : int;
  config const useCUGraph = false : bool;

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
  }
}
