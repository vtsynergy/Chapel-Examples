module EdgeCentric {
  use CSR;

  proc EC_Jaccard(type inType : unmanaged CSR, in graph : inType, type outType : real(?), outWeights : [] outType) {
    //Do stuff
    writeln("Edge Centric");
    writeln(inType : string);
    writeln(outType : string);
    assert(!inType.isWeighted, "Edge-centric does not support weighted input graphs");
  }
}
