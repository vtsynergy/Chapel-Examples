module EdgeCentric {
  use CSR;

  proc EC_Jaccard(type inType : unmanaged CSR, in inGraph : inType, type outType : unmanaged CSR(isWeighted = true), ref outGraph : outType) {
    writeln("Edge Centric");
    writeln(inType : string);
    writeln(outType : string);
    assert(!inType.isWeighted, "Edge-centric does not support weighted input graphs");

    //Kernels happen here
    
  }

  //FIXME: Should the intermediate steps be privatized?
  proc jaccard(in inCSR : CSR_handle, in outCSR : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) {
    //We only use inCSR here because outCSR is guaranteed to be weighted. inCSR may or may not be
    if (inCSR.desc.isWeighted) {
      //Recast both and call EC_Jaccard
      var inCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), inCSR);
      var outCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), outCSR);
      EC_Jaccard(inCSRInstance.type, inCSRInstance, outCSRInstance.type, outCSRInstance);
    } else {
      //Recast both and call EC_Jaccard
      var inCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = false, isVertexT64, isEdgeT64, isWeightT64), inCSR);
      //This will always be true to store the weights
      var outCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), outCSR);
      EC_Jaccard(inCSRInstance.type, inCSRInstance, outCSRInstance.type, outCSRInstance);
    }
  }

  proc jaccard(in inCSR : CSR_handle, in outCSR : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool) {
    if (outCSR.desc.isWeightT64) {
      jaccard(inCSR, outCSR, isVertexT64, isEdgeT64, false);
    } else {
      jaccard(inCSR, outCSR, isVertexT64, isEdgeT64, false);
    }
  }

  proc jaccard(in inCSR : CSR_handle, in outCSR : CSR_handle, param isVertexT64 : bool) {
    if (outCSR.desc.isEdgeT64) {
      jaccard(inCSR, outCSR, isVertexT64, false);
    } else {
      jaccard(inCSR, outCSR, isVertexT64, false);
    }
  }

  //This is the entrypoint, the user shouldn't need to interact with the intermediate steps of the ladder
  proc jaccard(in inCSR : CSR_handle, in outCSR : CSR_handle) {
    //TODO If the input format doesn't match the output, coerce it immediately and then use that instead
    //We compute in output-native format
    if (outCSR.desc.isVertexT64) {
      jaccard(inCSR, outCSR, false);
    } else {
      jaccard(inCSR, outCSR, false);
    }
  }
}
