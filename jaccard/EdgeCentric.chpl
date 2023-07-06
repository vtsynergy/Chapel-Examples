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
  use Time;

  proc EC_Jaccard(type csr_type : unmanaged CSR_arrays, in inGraph : CSR_base, inout outGraph : CSR_base, param isWeighted : bool) {
    writeln("Edge Centric");
    assert(!isWeighted, "Edge-centric does not support weighted input graphs");

    var fullInGraph = try! (inGraph : csr_type);
    var fullOutGraph = try! (outGraph : csr_type);
    
    //Kernels happen here
    //This is a trivial that just writes the thread ID for each edge. Should be easy to convert to EC_scan
    var gpu_region_time : stopwatch;
    var scan_time : stopwatch;
    var intersect_time : stopwatch;
    var weights_time : stopwatch;
    gpu_region_time.clear();
    gpu_region_time.start();
    on here.gpus[0] {
      //Declare device arrays, using element-types and sizes from the CSR objects for convenience
      var offsets: [fullOutGraph.offDom] fullOutGraph.offsets.eltType;
      var indices: [fullOutGraph.idxDom] fullOutGraph.indices.eltType;
	  var dests: [fullOutGraph.idxDom] fullOutGraph.indices.eltType;
      var weights: [fullOutGraph.weightDom] fullOutGraph.weights.eltType;
	  var jaccards: [fullOutGraph.weightDom] fullOutGraph.weights.eltType;
	  var trusses: [fullOutGraph.idxDom] fullOutGraph.indices.eltType;
	  var triangles: [fullOutGraph.idxDom] fullOutGraph.indices.eltType;
      //Copy data to device arrays
      offsets = fullInGraph.offsets;
      indices = fullInGraph.indices;
      //Do the kernel computation
      //This needs to be copied out of the `unmanaged CSR` instance to be usable in 1.29
      //If you use fullInGraph.numVerts directly in the if statement below, it appears to cancel most of the threads
      var offSize = fullInGraph.numVerts;
          scan_time.clear();
          scan_time.start();
          //Scan
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		  //forall i in (offsets.size - 1) {
		if (i>=0) && (i<=offSize){
		for j in offsets[i]..(offsets[i+1]-1)
		{
		 dests[j] = i:fullOutGraph.indices.eltType; ; 
			//weights[i] = i : fullOutGraph.weights.eltType; //The domain index space is probably integral, convert to whatever real(?) that weights needs to be
        }
		}
	  }
          scan_time.stop();
	  
	  //edge_t tid, i,  Ni, Nj, left, right, middle;
	  //vertex_t row, col;
	  //vertex_t ref, cur, ref_col, cur_col;
		
	 
          type idxType = int (if (fullOutGraph.iWidth == 64 || fullOutGraph.oWidth == 64) then 64 else 32);
	  //perform JS computations
          //Intersection
          intersect_time.clear();
          intersect_time.start();
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		
		var Ni : idxType; 
	    var Nj : idxType; 
	    var left : idxType; 
	    var right : idxType;  
	    var middle : idxType; 
		
	    var row : idxType; 
	    var col : idxType; 
	    var refs : idxType; 
	    var cur : idxType;
	    var ref_col : idxType;
	    var cur_col : idxType;
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
		
		
		
	 //var trusses: [fullOutGraph.offDom] fullOutGraph.offsets.eltType;
	 // var triangles: [fullOutGraph.offDom] fullOutGraph.offsets.eltType;
         // seperate kernel for the weights[tid]= weight_j[tid]/((weight_t)(Ni+Nj)-weight_j[tid]);
	  }
          intersect_time.stop();
	  
          //Weights
          weights_time.clear();
          weights_time.start();
	  forall i in indices.domain {
		assertOnGpu(); //Fail if this can't be GPU-ized
		var Ni : idxType; 
	    var Nj : idxType; 
	    
	    var row : idxType; 
	    var col : idxType; 
		
		row = indices[i];
		col = dests[i];
		Ni  = offsets[row + 1] - offsets[row];
		Nj  = offsets[col + 1] - offsets[col];
		triangles[i] = weights[i]:fullOutGraph.indices.eltType; 
		if (triangles[i] > 0){
			  trusses[i] = triangles[i] + 2;
		}
		jaccards[i]= weights[i]/(Ni + Nj - weights[i]);
		  //forall i in (offsets.size - 1) {
		
	  }
          weights_time.stop();
      //Copy arrays out
	  //dest_inds = dests;
      fullOutGraph.offsets = offsets;
      fullOutGraph.indices = indices;
      fullOutGraph.weights = jaccards;
	 // writeln("dests are ", dests);
    }
    gpu_region_time.stop();
    writeln("EC_Scan Elapsed (s): ", scan_time.elapsed());
    writeln("EC_Intersect Elapsed (s): ", intersect_time.elapsed());
    writeln("EC_Weights Elapsed (s): ", weights_time.elapsed());
    writeln("EC_GPU_Region Elapsed (s): ", gpu_region_time.elapsed());
	//writeln("offsets1 ", fullOutGraph.offsets[1]);
	//writeln("offsets2 ", fullOutGraph.offsets[2]);
    //for i in fullOutGraph.offsets[1]..fullOutGraph.offsets[2] do
	//	writeln("hello #", i);
	
	var count : fullOutGraph.indices.eltType; 
	for k in fullOutGraph.indices.domain{
	   if ( fullOutGraph.weights[k] > 0:fullOutGraph.weights.eltType){
		   count = count + 1;
	}}
	writeln(" Pairs with non zero intersection : ", count);
  }

  private proc jaccard(in inGraph : CSR_base, inout outGraph : CSR_base, param iWidth : int, param oWidth : int, param wWidth : int) {
    type arrType = unmanaged CSR_arrays(iWidth, oWidth, wWidth);
    if (inGraph.isWeighted) {
      EC_Jaccard(arrType, inGraph, outGraph, true);
    } else {
      EC_Jaccard(arrType, inGraph, outGraph, false);
    }
  }
  private proc jaccard(in inGraph : CSR_base, inout outGraph : CSR_base, param iWidth : int, param oWidth : int) {
    if (inGraph.isWeightT64) {
      jaccard(inGraph, outGraph, iWidth, oWidth, 64);
    } else {
      jaccard(inGraph, outGraph, iWidth, oWidth, 32);
    }
  }
  private proc jaccard(in inGraph : CSR_base, inout outGraph : CSR_base, param iWidth : int) {
    if (inGraph.isEdgeT64) {
      jaccard(inGraph, outGraph, iWidth, 64);
    } else {
      jaccard(inGraph, outGraph, iWidth, 32);
    }
  }
  //This is the entrypoint, the user shouldn't need to interact with the intermediate steps of the ladder
  proc jaccard(in inGraph : unmanaged CSR_base, inout outGraph : unmanaged CSR_base) {
    if (inGraph.isVertexT64) {
      jaccard(inGraph, outGraph, 64);
    } else {
      jaccard(inGraph, outGraph, 32);
    }
  }
}
