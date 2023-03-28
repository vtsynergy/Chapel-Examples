/*
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void jaccard_ec_scan(vertex_t n,
                           edge_t const *csrPtr,
                           vertex_t const *csrInd,
                           vertex_t *dest_ind)
{
  edge_t tid, i;
  tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
  if(tid<=n){
  
  //Ni=csrPtr[tid+1]-csrPtr[tid];
  for(i=csrPtr[tid];i<csrPtr[tid+1];i++)
  {
   dest_ind[i]=tid;
  }
  }
}*/

module EdgeCentric {
  use CSR;
  use GPU;

  proc EC_Jaccard(type inType : unmanaged CSR, in inGraph : inType, type outType : unmanaged CSR(isWeighted = true), ref outGraph : outType) {
    writeln("Edge Centric");
    writeln("inCSR before: ", inGraph);
    writeln("outCSR before: ", outGraph);
    assert(!inType.isWeighted, "Edge-centric does not support weighted input graphs");
    
    //Kernels happen here
    //This is a trivial that just writes the thread ID for each edge. Should be easy to convert to EC_scan
    on here.gpus[0] {
      //Declare device arrays, using element-types and sizes from the CSR objects for convenience
      var offsets: [outGraph.offDom] outGraph.offsets.eltType;
      var indices: [outGraph.idxDom] outGraph.indices.eltType;
	  var dests: [outGraph.idxDom] outGraph.indices.eltType;
      var weights: [outGraph.weightDom] outGraph.weights.eltType;
	  var jaccards: [outGraph.weightDom] outGraph.weights.eltType;
	  var trusses: [outGraph.idxDom] outGraph.indices.eltType;
	  var triangles: [outGraph.idxDom] outGraph.indices.eltType;
      //Copy data to device arrays
      offsets = inGraph.offsets;
      indices = inGraph.indices;
      //Do the kernel computation
      //This needs to be copied out of the `unmanaged CSR` instance to be usable in 1.29
      //If you use inGraph.numVerts directly in the if statement below, it appears to cancel most of the threads
      var offSize = inGraph.numVerts;
          //Scan
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		  //forall i in (offsets.size - 1) {
		if (i>=0) && (i<=offSize){
		for j in offsets[i]..(offsets[i+1]-1)
		{
		 dests[j] = i:outGraph.indices.eltType; ; 
			//weights[i] = i : outGraph.weights.eltType; //The domain index space is probably integral, convert to whatever real(?) that weights needs to be
        }
		}
	  }
	  
	  //edge_t tid, i,  Ni, Nj, left, right, middle;
	  //vertex_t row, col;
	  //vertex_t ref, cur, ref_col, cur_col;
		
	 
	  //perform JS computations
          //Intersection
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		
		var Ni : outGraph.indices.eltType; 
	    var Nj : outGraph.indices.eltType; 
	    var left : outGraph.indices.eltType; 
	    var right : outGraph.indices.eltType;  
	    var middle : outGraph.indices.eltType; 
		
	    var row : outGraph.offsets.eltType; 
	    var col : outGraph.offsets.eltType; 
	    var refs : outGraph.offsets.eltType; 
	    var cur : outGraph.offsets.eltType;
	    var ref_col : outGraph.offsets.eltType;
	    var cur_col : outGraph.offsets.eltType;
		//edge_t tid, i,  Ni, Nj, left, right, middle;
		//vertex_t row, col;
		//vertex_t ref, cur, ref_col, cur_col;
		
		row = indices[i];
		col = dests[i];
		Ni  = offsets[row + 1] - offsets[row];
		Nj  = offsets[col + 1] - offsets[col];
		//ref = (Ni < Nj) ? row : col;
		refs = col ;
		if (Ni <Nj){
		    refs = row ;
		}
		cur = row;
		//cur = (Ni < Nj) ? col : row;
		if (Ni <Nj){  
			cur = col ;
		}
		// col = csrInd[j];
        // compute new intersection weights
        // search for the element with the same column index in the reference row
		// for (i = csrPtr[ref] ; i < csrPtr[ref + 1]; i ++) {
		
		for j in offsets[refs]..(offsets[refs+1]-1){
			ref_col = indices[j];
			// binary search (column indices are sorted within each row)
			left  = offsets[cur];
			right = offsets[cur + 1] - 1;
			while (left <= right) {
			  middle = (left + right) >> 1;
			  cur_col       = indices[middle];
			  if (cur_col > ref_col) {
				right = middle - 1;
			  } else if (cur_col < ref_col) {
				left = middle + 1;
			  } else {
				weights[i] = weights[i]+1;
				break;
			  }
			}
	    }
		
		
		
	 //var trusses: [outGraph.offDom] outGraph.offsets.eltType;
	 // var triangles: [outGraph.offDom] outGraph.offsets.eltType;
         // seperate kernel for the weights[tid]= weight_j[tid]/((weight_t)(Ni+Nj)-weight_j[tid]);
	  }
	  
          //Weights
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		var Ni : outGraph.indices.eltType; 
	    var Nj : outGraph.indices.eltType; 
	    
	    var row : outGraph.offsets.eltType; 
	    var col : outGraph.offsets.eltType; 
		
		row = indices[i];
		col = dests[i];
		Ni  = offsets[row + 1] - offsets[row];
		Nj  = offsets[col + 1] - offsets[col];
		triangles[i] = weights[i]:outGraph.indices.eltType; 
		if (triangles[i] > 0){
			  trusses[i] = triangles[i] + 2;
		}
		jaccards[i]= weights[i]/(Ni + Nj - weights[i]);
		  //forall i in (offsets.size - 1) {
		
	  }
      //Copy arrays out
	  //dest_inds = dests;
      outGraph.offsets = offsets;
      outGraph.indices = indices;
      outGraph.weights = jaccards;
	 // writeln("dests are ", dests);
    }
	//writeln("offsets1 ", outGraph.offsets[1]);
	//writeln("offsets2 ", outGraph.offsets[2]);
    //for i in outGraph.offsets[1]..outGraph.offsets[2] do
	//	writeln("hello #", i);
	
	writeln("inCSR after: ", inGraph);
    writeln("outCSR after: ", outGraph);
	var count : outGraph.indices.eltType; 
	for k in outGraph.indices.domain{
	   if ( outGraph.weights[k] > 0:outGraph.weights.eltType){
		   count = count + 1;
	}}
	writeln(" Pairs with non zero intersection : ", count);
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
