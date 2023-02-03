module CSR {
  //Need to implement a record for the CSRv2 header format
  // This might be a pain because it doesn't look like Chapel has bitfields, so we will need to create a bitmask enum to make operations simpler

  //Datatypes to support file read
  enum CSR_header_flags {
  // TODO give uint initializations of these
    isWeighted = 1 << 63,
    isZeroIndexed = 1 << 62,
    isDirected = 1 << 61,
    hasReverseEdges = 1 << 60,
    isVertexT64 = 1 << 59,
    isEdgeT64 = 1 << 58,
    isWeightT64 = 1 << 57,
  };

  record CSR_file_header {
    var binaryFormatVersion : int(64) = 1;
    var numVerts : int(64) = 0;
    var numEdges : int(64) = 0;
    //Because of how Chapel casts to enums we can't store "all false (0)" or ORed values in an enum, so the flags field has to be treated as int(64)
    var flags : int(64) = 0;
  }

  //Can we make this a generic type to accept both 32- and 64-bit vertices/edges/weights?
  record CSR {
    var numEdges : int(64);
    var numVerts : int(64);
    param isWeighted : bool;
    var isZeroIndexed : bool;
    var isDirected : bool;
    var hasReverseEdges : bool;
    param isVertexT64 : bool;
    param isEdgeT64 : bool;
    param isWeightT64 : bool;
    var idxDom : domain(1) = {1..numEdges};
    var indices : [idxDom] int(if isVertexT64 then 64 else 32);
    var offDom : domain(1) = {1..(numVerts+1)};
    var offsets : [offDom] int(if isEdgeT64 then 64 else 32);
    var weightDom : domain(1) = {1..(if isWeighted then numEdges else 0)}; //Degenerate if we don't have weights
    var weights : [idxDom] int(if isWeightT64 then 64 else 32);
 //   var offsets : [1..numVerts+1] int(?);
 //   var weights : [1..numEdges] real(?);
  }

  //Need to read CSRv2-formatted data
  //FIXME, idiomatically, can we use type reflection to infer/coerce the size of the counting variables?
  proc parseCSRHeader(header : CSR_file_header, ref binFmtVers : int(64), ref numVerts : int(64), ref numEdges : int(64), ref isWeighted : bool, ref isZeroIndexed : bool, ref isDirected : bool, ref hasReverseEdges : bool, ref isVertexT64 : bool, ref isEdgeT64 : bool, ref isWeightT64 : bool) {
    //Directly map the counting variables, using coersion if necessary
    //Bitmask the flags field
    if ((header.flags & CSR_header_flags.isWeighted) != 0) { isWeighted = true; }
    if ((header.flags & CSR_header_flags.isZeroIndexed) != 0) { isZeroIndexed = true; }
    if ((header.flags & CSR_header_flags.isDirected) != 0) { isDirected = true; }
    if ((header.flags & CSR_header_flags.hasReverseEdges) != 0) { hasReverseEdges = true; }
    if ((header.flags & CSR_header_flags.isVertexT64) != 0) { isVertexT64 = true; }
    if ((header.flags & CSR_header_flags.isEdgeT64) != 0) { isEdgeT64 = true; }
    if ((header.flags & CSR_header_flags.isWeightT64) != 0) { isWeightT64 = true; }
  }

  proc readCSRFile(inFile : string): CSR {
    var myCSR : CSR;
    //Set up a file reader, however that is done
    //Read the fixed-size header
    var header : CSR_file_header;
    //Parse it
    var binFmtVers : int(64);
    parseCSRHeader(header, binFmtVers, myCSR.numVerts, myCSR.numEdges, myCSR.isWeighted, myCSR.isZeroIndexed);
    //Assert that the binary format version is the one we're expecting (Vers. 2)

    return myCSR;
  }


  //Need to write CSRv2-formatted data
  //FIXME, idiomatically, can we use type reflection to infer/coerce the size of the counting variables?
  proc buildCSRHeader(binFmtVers : int(64), numVerts : int(64), numEdges : int(64), isWeighted : bool, isZeroIndexed : bool, isDirected : bool, hasReverseEdges : bool, isVertexT64 : bool, isEdgeT64 : bool, isWeightT64 : bool): CSR_file_header {
    //
  }

}
