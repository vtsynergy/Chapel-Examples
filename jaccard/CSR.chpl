//Has to be prototype because the IO calls throw. Apparently
prototype module CSR {
  use IO; //Need for file ops
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
  }
//The new parser returns the elaborated CSR type, so that the application can use it directly to construct
//proc parseCSRHeader(in header : CSR_file_header, out binFmtVers : int(64), out numVerts : int(64), out numEdges : int(64), out isWeighted : bool, out isZeroIndexed : bool, out isDirected : bool, out hasReverseEdges : bool, out isVertexT64 : bool, out isEdgeT64: bool, ref isWeightT64: bool) : type {
/*proc parseCSRHeader(in header : CSR_file_header) type {
  if ((header.flags & (CSR_header_flags.isWeighted : int(64))) != 0) { //weighted branch
    return CSR(isWeighted = false, isVertexT64 = false, isEdgeT64 = false, isWeightT64 = false);
  } else { //Unweighted branch
    return CSR(isWeighted = true, isVertexT64 = false, isEdgeT64 = false, isWeightT64 = false);
  }
  
  /*
    if ( { isWeighted = true; }
    if ((header.flags & CSR_header_flags.isZeroIndexed) != 0) { isZeroIndexed = true; }
    if ((header.flags & CSR_header_flags.isDirected) != 0) { isDirected = true; }
    if ((header.flags & CSR_header_flags.hasReverseEdges) != 0) { hasReverseEdges = true; }
    if ((header.flags & CSR_header_flags.isVertexT64) != 0) { isVertexT64 = true; }
    if ((header.flags & CSR_header_flags.isEdgeT64) != 0) { isEdgeT64 = true; }
    if ((header.flags & CSR_header_flags.isWeightT64) != 0) { isWeightT64 = true; }
  type myType = CSR;
  writeln(myType : string);
  myType = CSR(false);
  writeln(myType : string);
  return myType;
 */
}
*/
proc readCSRArrays(in readChannel, type CSR_spec): CSR_spec {
  //  var myCSR = new CSR_spec();
  
}

proc CSRUser(in inFile : string) {
    ///File operations (which they are reworking as of 1.29.0)
    //FIXME: Add error handling
    //FIXME: Reimplement using readThis methods
    //Open
    var readFile = IO.open(inFile, IO.iomode.r);
    //Create a read channel
    var readChannel = readFile.reader(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
    //Read the fixed-size header

    //  var header = {0, 0, 0, 0} : CSR_file_header; // "illegal cast from DefaultAssociativeDom(int(64),true) to CSR_file_header" // so I guess don't initialize here?
    var header : CSR_file_header;
    var expectedBinFmt = header.binaryFormatVersion; //FIXME I can't figure out a better way to grab the default integral constant from the record type, other than just copying it from an entity that has been default initialized
    readChannel.read(header);
    //readChannel.read(header.binaryFormatVersion);
    writeln(header);
    //Assert that the binary format version is the one we're expecting (Vers. 2)
    
assert(header.binaryFormatVersion == expectedBinFmt, "Binary version of ", inFile, " is ", header.binaryFormatVersion, " but expected ", expectedBinFmt);
  /*
    type myCSRType = parseCSRHeader(header);
    writeln(myCSRType :string);
    var h2 = header;
    h2.flags = CSR_header_flags.isWeighted : int(64);
    writeln(parseCSRHeader(h2) : string);
  */
//    var myCSR = readCSRArrays(readChannel, myCSRType);
}
//I couldn't figure out how to do the equivalent of gradually-nesting specializations so here we are, a 16-way split for the 4 boolean bits
//proc CSRFactory(numEdges : int(64), numVerts : int(64), isWeighted : bool, isZeroIndexed : bool, isDirected : bool, hasReverseEdges : bool) : CSR {
/* proc CSRFactory(numEdges : int(64), numVerts : int(64), isWeighted : bool, isZeroIndexed : bool, isDirected : bool, hasReverseEdges : bool) :CSR(isWeighted = true) where isWeighted == true {
    return new CSR(numEdges, numVerts, true, isZeroIndexed, isDirected, hasReverseEdges);
}
proc CSRFactory(numEdges : int(64), numVerts : int(64), isWeighted : bool, isZeroIndexed : bool, isDirected : bool, hasReverseEdges : bool) :CSR(isWeighted = false) where isWeighted == false {
    return new CSR(numEdges, numVerts, false, isZeroIndexed, isDirected, hasReverseEdges);
}*/
/*
OLD HEADER PARSE
  //Need to read CSRv2-formatted data
  //FIXME, idiomatically, can we use type reflection to infer/coerce the size of the counting variables?
  proci parseCSRHeader(header : CSR_file_header, ref binFmtVers : int(64), ref numVerts : int(64), ref numEdges : int(64), ref isWeighted : bool, ref isZeroIndexed : bool, ref isDirected : bool, ref hasReverseEdges : bool, ref isVertexT64 : bool, ref isEdgeT64 : bool, ref isWeightT64 : bool) {
    //Directly map the counting variables, using coersion if necessary
    //Bitmask the flags field
  }


  proc readCSRFile(inFile : string): CSR {
  //proc readCSRFile(inFile : string): void {

    ///File operations (which they are reworking as of 1.29.0)
    //FIXME: Add error handling
    //FIXME: Reimplement using readThis methods
    //Open
    var readFile = IO.open(inFile, IO.iomode.r);
    //Create a read channel
    var readChannel = readFile.reader(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
    //Read the fixed-size header

    //  var header = {0, 0, 0, 0} : CSR_file_header; // "illegal cast from DefaultAssociativeDom(int(64),true) to CSR_file_header" // so I guess don't initialize here?
    var header : CSR_file_header;
    var expectedBinFmt = header.binaryFormatVersion; //FIXME I can't figure out a better way to grab the default integral constant from the record type, other than just copying it from an entity that has been default initialized
    readChannel.read(header);
    //readChannel.read(header.binaryFormatVersion);
    writeln(header);
    //Assert that the binary format version is the one we're expecting (Vers. 2)
    
assert(header.binaryFormatVersion == expectedBinFmt, "Binary version of ", inFile, " is ", header.binaryFormatVersion, " but expected ", expectedBinFmt);

    //We can't even construct the CSR return record without having the header for typing information
/*    type csr_vert;
    type csr_edge;
    type csr_weight;
    if (header.flags & (CSR_header_flags.isVertexT64 : int(64)) == 0) {
      csr_vert = int(32);
    } else {
      csr_vert = int(64);
    }
    if (header.flags & CSR_header_flags.isEdgeT64 == 0) {
      csr_edge = int(32);
    } else {
      csr_edge = int(64);
    }
    if (header.flags & CSR_header_flags.isWeightT64 == 0) {
      csr_weight = int(32);
    } else {
      csr_weight = int(64);
    }
    var myCSR : CSR(indices = csr_vert, offsets = csr_edge, weights = csr_weight);
*/
/*    return CSRFactory(
      numEdges = header.numEdges,
      numVerts = header.numVerts,
      isWeighted = (header.flags & (CSR_header_flags.isWeighted : int(64)) != 0),
      isZeroIndexed = (header.flags & (CSR_header_flags.isZeroIndexed : int(64)) != 0),
      isDirected = (header.flags & (CSR_header_flags.isDirected : int(64)) != 0),
      hasReverseEdges = (header.flags & (CSR_header_flags.hasReverseEdges : int(64)) != 0)
    );
*/
/*    
    return new CSR(
      numEdges = header.numEdges,
      numVerts = header.numVerts,
      isWeighted = (header.flags & (CSR_header_flags.isWeighted : int(64)) != 0),
      isZeroIndexed = (header.flags & (CSR_header_flags.isZeroIndexed : int(64)) != 0),
      isDirected = (header.flags & (CSR_header_flags.isDirected : int(64)) != 0),
      hasReverseEdges = (header.flags & (CSR_header_flags.hasReverseEdges : int(64)) != 0)
    );
  */
    if ( (header.flags & (CSR_header_flags.isWeighted : int(64)) != 0) ) {
      return new CSR(
        numEdges = header.numEdges,
        numVerts = header.numVerts,
        isWeighted = true,
        isZeroIndexed = (header.flags & (CSR_header_flags.isZeroIndexed : int(64)) != 0),
        isDirected = (header.flags & (CSR_header_flags.isDirected : int(64)) != 0),
        hasReverseEdges = (header.flags & (CSR_header_flags.hasReverseEdges : int(64)) != 0)
       );
    } else {
      return new CSR(
        numEdges = header.numEdges,
        numVerts = header.numVerts,
        isWeighted = false,
        isZeroIndexed = (header.flags & (CSR_header_flags.isZeroIndexed : int(64)) != 0),
        isDirected = (header.flags & (CSR_header_flags.isDirected : int(64)) != 0),
        hasReverseEdges = (header.flags & (CSR_header_flags.hasReverseEdges : int(64)) != 0)
      );
    }

/*    return new CSR(
      numEdges = header.numEdges,
      numVerts = header.numVerts,
      isWeighted = (header.flags & (CSR_header_flags.isWeighted : int(64)) != 0),
      isZeroIndexed = (header.flags & (CSR_header_flags.isZeroIndexed : int(64)) != 0),
      isDirected = (header.flags & (CSR_header_flags.isDirected : int(64)) != 0),
      hasReverseEdges = (header.flags & (CSR_header_flags.hasReverseEdges : int(64)) != 0),
      isVertexT64 = (header.flags & (CSR_header_flags.isVertexT64 : int(64)) != 0),
      isEdgeT64 = (header.flags & (CSR_header_flags.isEdgeT64 : int(64)) != 0),
      isWeightT64 = (header.flags & (CSR_header_flags.isEdgeT64 : int(64)) != 0)
    );
*/
    //return new CSR(0, 0, false, false, false, false, false, false, false, {0}: int(32), {0}: int(32), {0.0} : real(32));
  }
*/


  //Need to write CSRv2-formatted data
  //FIXME, idiomatically, can we use type reflection to infer/coerce the size of the counting variables?
//  proc buildCSRHeader(binFmtVers : int(64), numVerts : int(64), numEdges : int(64), isWeighted : bool, isZeroIndexed : bool, isDirected : bool, hasReverseEdges : bool, isVertexT64 : bool, isEdgeT64 : bool, isWeightT64 : bool): CSR_file_header {
    //
 // }

}
