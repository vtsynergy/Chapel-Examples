module Jaccard {
  use CSR;
  use EdgeCentric;
  use CuGraph;

  //TODO Map the "ENABLE" and "DISABLE" preprocessor behavior from C++ to `config param`

  //Need to expose device selection to the command line

  //Need to write a separate bridge module to the SYCL implementations (far later)

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
    var inBase = readCSRFileToBase(inFile);
    //Create an empty output CSR of the same type as the input, the kernel pipelines will populate it, but always need weights on outputs, that's where we store JS values
    var outBase = MakeCSR(new CSR_base(
      numEdges = inBase.numEdges,
      numVerts = inBase.numVerts,
      isWeighted = true,
      isZeroIndexed = inBase.isZeroIndexed,
      isDirected = inBase.isDirected,
      hasReverseEdges = inBase.hasReverseEdges,
      isVertexT64= inBase.isVertexT64,
      isEdgeT64 = inBase.isEdgeT64,
      isWeightT64 = inBase.isWeightT64
    ));
    //Launch the selected kernel pipeline
    if (useCUGraph) {
      CuGraph.jaccard(inBase, outBase);
    } else {
      EdgeCentric.jaccard(inBase, outBase);
    }
    //Write the output file
    writeCSRFile(outFile, outBase);
  }
}
