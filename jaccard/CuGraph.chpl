module CuGraph {
  use CSR;
  use GPU;
  param MAX_GPU_BLOCKS=33554432;
//  param MAX_GPU_BLOCKS=65535;

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
      //RowSum
      var partialSum: [offsets.domain.interior(-inGraph.numVerts)] outGraph.weights.eltType; //We only want the first numVerts elements starting at 0 (not the element at numVerts)
      forall x in partialSum.domain {
        var start = offsets[x];
        var end = offsets[x+1];
        if (inType.isWeighted) {
          assert(false, "Weighted RowSum not yet implemented");
        } else {
          assertOnGpu(); //Fail if this can't be GPU-ized
          partialSum[x] = end-start;
        }
      }
      //Fill is trivial
      var intersectWeight: [outGraph.weightDom] atomic outGraph.weights.eltType;
      var neighborSum: [outGraph.weightDom] outGraph.weights.eltType;
      intersectWeight = 0.0 : outGraph.weights.eltType;
      //Intersection
      param isYBlock = 4;
      param isXBlock = 32 / isYBlock;
      param isZBlock = 8;
      var isXGrid = 1;
      var isYGrid = 1;
      var isZGrid = min((inGraph.numVerts + isZBlock -1)/isZBlock, MAX_GPU_BLOCKS);
      //Each of these double-loop lines is using the forall to define the CUDA grid/block dimensions, and the for to do the corresponding intra-thread loop
      forall z in 0..<isZGrid*isZBlock {
      for row in z..<inGraph.numVerts by isZGrid*isZBlock {//Rows/Z
        forall y in 0..<isYGrid*isYBlock {
        for j in (offsets[row]+y)..<offsets[row+1] by isYGrid*isYBlock {  //offsets[row..row+1]/Y
          var col = indices[y];
          // find which row has least elements (and call it reference row)
          var Ni = offsets[row+1] - offsets[row];
          var Nj = offsets[col+1] - offsets[col];
          var refer = if (Ni < Nj) then row else col;
          var cur = if (Ni < Nj) then col else row;

          // compute new sum weights
          neighborSum[j] = partialSum[row] + partialSum[col];
          
          //compute new intersection weights
          // search for the element with the same column index in the reference row
          forall x in 0..<isXGrid*isXBlock {
          for i in (offsets[refer]+x)..<offsets[refer+1] by isXGrid*isXBlock { // offsets[ref..ref+1] / Z
            var match = -1 : outGraph.indices.eltType;
            var ref_col = indices[i];
            var ref_val : outGraph.weights.eltType;
            if (inGraph.isWeighted) {
              assert(false, "Weighted VC Intersectoin not yet implemented");
              //TODO ref_val = inGraph.weights[ref_col];
            } else {
              ref_val = 1.0;
            }

            //binary search (column indices are sorted within each row)
            var left = offsets[cur];
            var right = offsets[cur+1]-1;
            while (left <= right) {
              var middle = (left + right) >> 1;
              var cur_col = indices[middle];
              if (cur_col > ref_col) {
                right = middle-1;
              } else if (cur_col < ref_col) {
                left = middle+1;
              } else {
                match = middle;
                break;
              }
            }

            //If the element with the same column index in the reference row has been found
            if (match != -1) {
              //TODO, do we need an order qualifier?
              intersectWeight[j].add(ref_val);
            }
          } } //close 'z' forall and 'i' for loops
        } } //close 'y' forall and 'j' for loops
      } } //close 'z' forall and 'row' for loops
      
      //JaccardWeights
      forall x in weights.domain {
        //FIXME, could an order qualifer give better performance?
        var Wi = intersectWeight[x].read();
        var Ws = neighborSum[x];
        var Wu = Ws - Wi;
        weights[x] = (Wi / Wu);
      }
      //Copy arrays out
      //FIXME, once we can coerce on the host, no need to write offsets/indices here
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
