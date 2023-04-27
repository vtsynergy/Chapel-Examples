module CuGraph {
  use CSR;
  use GPU;
  use Time;

  param MAX_GPU_BLOCKS=33554432;
//  param MAX_GPU_BLOCKS=65535;

  //Mix some CUDA and SYCL abstractions
  //For tuples, X=0, Y=1, Z=2
  record nd_item {
    var global_id : 3*int(64);
    var block_id : 3*int(64);
    var thread_id : 3*int(64);
    var global_dim : 3*int(64);
    var grid_dim : 3*int(64);
    var block_dim : 3*int(64);
  }

  //Just a shorthand to de-linearize the ID
  //For tuples, X=0, Y=1, Z=2
  proc get_ND_ID(in gridDim : 3*int(64), in blockDim : 3*int(64), in linearID : int(64)) : nd_item {
    var ret : nd_item;
    ret.block_dim = blockDim;
    ret.grid_dim = gridDim;
    ret.global_dim = ((blockDim(0)*gridDim(0)), (blockDim(1)*gridDim(1)), (blockDim(2)*gridDim(2)));
    var inBlockLinear = linearID % (blockDim(0)*blockDim(1)*blockDim(2));
    var blockLinear = linearID / (blockDim(0)*blockDim(1)*blockDim(2)) : int(64);
    ret.thread_id(0) = inBlockLinear % blockDim(0);
    ret.thread_id(1) = (inBlockLinear / blockDim(0)) : int(64) % (blockDim(1));
    ret.thread_id(2) = (inBlockLinear / (blockDim(0) * blockDim(1))) : int(64);
    ret.block_id(0) = blockLinear % gridDim(0);
    ret.block_id(1) = (blockLinear / gridDim(0)) : int(64) % (gridDim(1));
    ret.block_id(2) = (blockLinear / (gridDim(0) * gridDim(1))) : int(64);
    ret.global_id(0) = ret.thread_id(0) + ret.block_id(0)*blockDim(0);
    ret.global_id(1) = ret.thread_id(1) + ret.block_id(1)*blockDim(1);
    ret.global_id(2) = ret.thread_id(2) + ret.block_id(2)*blockDim(2);
    return ret;
  }

  proc VC_Jaccard(type inType : unmanaged CSR, in inGraph : inType, type outType : unmanaged CSR(isWeighted = true), ref outGraph : outType) {
    //Do stuff
    writeln("Vertex Centric");
    assert(!inType.isWeighted, "Vertex-centric weighted input support not yet implemented");

    //Kernels happen here
    //This is a trivial that just writes the thread ID for each edge
    var gpu_region_time : stopwatch;
    var rowsum_time : stopwatch;
    var fill_time : stopwatch;
    var intersect_time : stopwatch;
    var weights_time : stopwatch;
    gpu_region_time.clear();
    gpu_region_time.start();
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
      rowsum_time.clear();
      rowsum_time.start();
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
      rowsum_time.stop();
      //Fill is trivial
      var intersectWeight: [outGraph.weightDom] atomic outGraph.weights.eltType;
      var neighborSum: [outGraph.weightDom] outGraph.weights.eltType;
      fill_time.clear();
      fill_time.start();
      intersectWeight = 0.0 : outGraph.weights.eltType;
      fill_time.stop();
      //Intersection
      param isYBlock = 4;
      param isXBlock = 32 / isYBlock;
      param isZBlock = 8;
      var isXGrid = 1;
      var isYGrid = 1;
      var isZGrid = min((inGraph.numVerts + isZBlock -1)/isZBlock, MAX_GPU_BLOCKS);

      //Yanked from EdgeCentric
      //If you use inGraph.numVerts directly in the if statement below, it appears to cancel most of the threads
      var offSize = inGraph.numVerts;

      //As of 1.30, only 1D foralls are supported, need to convert 3D->linear->3D threads
      var workSize : int(64) = ((isXBlock*isXGrid)*(isYBlock*isYGrid)*(isZBlock*isZGrid));
      intersect_time.clear();
      intersect_time.start();
      forall linear_id in 0..<workSize {
      var tid = get_ND_ID((isXGrid, isYGrid, isZGrid), (isXBlock, isYBlock, isZBlock), linear_id);
      //Each of these double-loop lines is using the forall to define the CUDA grid/block dimensions, and the for to do the corresponding intra-thread loop
      //forall z in 0..<isZGrid*isZBlock {
      //Since by clauses are breaking GPU-ization in 1.30, replace with whiles
      var row = tid.global_id(2) : outGraph.offsets.eltType;
      while (row < offSize) {
      //for row in (tid.global_id(2))..<inGraph.numVerts by tid.global_dim(2) {//Rows/Z
        //forall y in 0..<isYGrid*isYBlock {
        var j = offsets[row]+tid.global_id(1);
        while (j < offsets[row+1]) {
        //for j in (offsets[row]+tid.global_id(1))..<offsets[row+1] by tid.global_dim(1) {  //offsets[row..row+1]/Y
          var col = indices[j];
          // find which row has least elements (and call it reference row)
          var Ni = offsets[row+1] - offsets[row];
          var Nj = offsets[col+1] - offsets[col];
          var refer = if (Ni < Nj) then row else col;
          var cur = if (Ni < Nj) then col else row;

          // compute new sum weights
          neighborSum[j] = partialSum[row] + partialSum[col];
          
          //compute new intersection weights
          // search for the element with the same column index in the reference row
          //forall x in 0..<isXGrid*isXBlock {
          var i = offsets[refer]+tid.global_id(0);
          while (i < offsets[refer+1]) {
          //for i in (offsets[refer]+tid.global_id(0))..<offsets[refer+1] by tid.global_dim(0) { // offsets[ref..ref+1] / Z
            assertOnGpu(); //Fail if this can't be GPU-ized
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
          i += tid.global_dim(0);
          } //} //close 'z' forall and 'i' for loops
        j += tid.global_dim(1);
        } //} //close 'y' forall and 'j' for loops
      row += tid.global_dim(2) : outGraph.offsets.eltType;
      } } //close 'z' forall and 'row' for loops
      intersect_time.stop();
      
      //JaccardWeights
      weights_time.clear();
      weights_time.start();
      forall x in weights.domain {
          //assertOnGpu(); //Fail if this can't be GPU-ized
        //FIXME, could an order qualifer give better performance?
        var Wi = intersectWeight[x].read();
        var Ws = neighborSum[x];
        var Wu = Ws - Wi;
        weights[x] = (Wi / Wu);
      }
      weights_time.stop();
      //Copy arrays out
      //FIXME, once we can coerce on the host, no need to write offsets/indices here
      outGraph.offsets = offsets;
      outGraph.indices = indices;
      outGraph.weights = weights;
    }
    gpu_region_time.stop();
    writeln("VC_RowSum Elapsed (s): ", rowsum_time.elapsed());
    writeln("VC_Fill Elapsed (s): ", fill_time.elapsed());
    writeln("VC_Intersect Elapsed (s): ", intersect_time.elapsed());
    writeln("VC_Weights Elapsed (s): ", weights_time.elapsed());
    writeln("VC_GPU_Region Elapsed (s): ", gpu_region_time.elapsed());
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
