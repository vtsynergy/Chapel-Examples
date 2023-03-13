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

    //Read the input file and set up host arrays (use generic methods to support different FP types)
    var isZeroIndexed : bool;
    var isDirected : bool;
    var hasReverseEdges : bool;
    var inCSR = readCSRFile(inFile, isZeroIndexed, isDirected, hasReverseEdges);
    //Create an empty output CSR of the same type as the input, the kernel pipelines will populate it
    var outCSR = MakeCSR(isWeighted = true, isVertexT64 = inCSR.desc.isVertexT64, isEdgeT64 = inCSR.desc.isEdgeT64, isWeightT64 = inCSR.desc.isWeightT64, inCSR.desc.numEdges, inCSR.desc.numVerts);
    //Launch the selected kernel pipeline
    if (useCUGraph) {
      CuGraph.jaccard(inCSR, outCSR);
    } else {
      EdgeCentric.jaccard(inCSR, outCSR);
    }
    //Write the output file
    writeCSRFile(outFile, inCSR, isZeroIndexed, isDirected, hasReverseEdges); //TODO replace with outCSR once we have a copy operator for CSR handles
  }
}
