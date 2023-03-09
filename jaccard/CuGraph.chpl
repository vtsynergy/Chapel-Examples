module CuGraph {
  use CSR;
  use GPU;


  proc VC_Jaccard(type inType : unmanaged CSR, in inGraph : inType, type outType : unmanaged CSR(isWeighted = true), ref outGraph : outType) {
    //Do stuff
    writeln("Vertex Centric");
    writeln(inType : string);
    writeln(outType : string);
    assert(!inType.isWeighted, "Vertex-centric weighted input support not yet implemented");

    //Kernels happen here
    //This is a trivial that just writes the thread ID for each edge
    on here.gpus[0] {
      //Declare device arrays, using element-types and sizes from the CSR objects for convenience
      var offsets: [outGraph.offDom] outGraph.offsets.eltType;
      var indices: [outGraph.idxDom] outGraph.indices.eltType;
      var weights: [outGraph.weightDom] outGraph.weights.eltType;
      //Copy data to device arrays
      offsets = inGraph.offsets;
      indices = inGraph.indices;
      //Do the kernel computation
      forall i in indices.domain {
        assertOnGpu(); //Fail if this can't be GPU-ized
        weights[i] = i : outGraph.weights.eltType; //The domain index space is probably integral, convert to whatever real(?) that weights needs to be
      }
      //Copy arrays out
      outGraph.offsets = offsets;
      outGraph.indices = indices;
      outGraph.weights = weights;
    }
    writeln("inCSR after: ", inGraph);
    writeln("outCSR after: ", outGraph);
  }

  //FIXME: Should the intermediate steps be privatized?
  proc jaccard(in inCSR : CSR_handle, in outCSR : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) {
    //We only use inCSR here because outCSR is guaranteed to be weighted. inCSR may or may not be
    if (inCSR.desc.isWeighted) {
      //Recast both and call VC_Jaccard
      var inCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), inCSR);
      var outCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), outCSR);
      VC_Jaccard(inCSRInstance.type, inCSRInstance, outCSRInstance.type, outCSRInstance);
    } else {
      //Recast both and call VC_Jaccard
      var inCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = false, isVertexT64, isEdgeT64, isWeightT64), inCSR);
      //This will always be true to store the weights
      var outCSRInstance = ReinterpretCSRHandle(unmanaged CSR(isWeighted = true, isVertexT64, isEdgeT64, isWeightT64), outCSR);
      VC_Jaccard(inCSRInstance.type, inCSRInstance, outCSRInstance.type, outCSRInstance);
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
