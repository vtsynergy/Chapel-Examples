module CuGraph {
  use CSR;

  proc VC_Jaccard(type inType : CSR, in graph : inType, type outType : real(?), outWeights : [] outType) {
    //Do stuff
    writeln("Vertex Centric");
    writeln(inType : string);
    writeln(outType : string);
    assert(!inType.isWeighted, "Vertex-centric weighted input support not yet implemented");
  }

}
