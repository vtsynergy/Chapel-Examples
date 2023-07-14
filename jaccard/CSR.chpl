//Has to be prototype because the IO calls throw. Apparently
prototype module CSR {
  use IO; //Need for file ops
  use CTypes; //Need for handle's void pointer

  //Datatypes to support file read
  enum CSR_header_flags {
  // C++ on linux seems to initialize the bits from LSB to MSB
    isWeighted = 1 << 0,
    isZeroIndexed = 1 << 1,
    isDirected = 1 << 2,
    hasReverseEdges = 1 << 3,
    isVertexT64 = 1 << 4,
    isEdgeT64 = 1 << 5,
    isWeightT64 = 1 << 6,
  };

  private param CSR_BINARY_FORMAT_VERSION : int(64) = 2;
  record CSR_file_header {
    var binaryFormatVersion : int(64) = CSR_BINARY_FORMAT_VERSION;
    var numVerts : int(64) = 0;
    var numEdges : int(64) = 0;
    //Because of how Chapel casts to enums we can't store "all false (0)" or ORed values in an enum, so the flags field has to be treated as int(64)
    var flags : int(64) = 0;
    proc init() { }
    proc init=(rhs : CSR_base) {
      this.numVerts = rhs.numVerts;
      this.numEdges = rhs.numEdges;
      //flags
      this.flags = 0; //Have to start with a non-compound initialization
      if (rhs.isWeighted) { this.flags |= (CSR_header_flags.isWeighted : int(64)); }
      if (rhs.isZeroIndexed) { this.flags |= (CSR_header_flags.isZeroIndexed : int(64)); }
      if (rhs.isDirected) { this.flags |= (CSR_header_flags.isDirected : int(64)); }
      if (rhs.hasReverseEdges) { this.flags |= (CSR_header_flags.hasReverseEdges : int(64)); }
      if (rhs.isVertexT64) { this.flags |= (CSR_header_flags.isVertexT64 : int(64)); }
      if (rhs.isEdgeT64) { this.flags |= (CSR_header_flags.isEdgeT64 : int(64)); }
      if (rhs.isWeightT64) { this.flags |= (CSR_header_flags.isWeightT64 : int(64)); }
    }
    operator =(ref lhs: CSR_file_header, rhs : CSR_base) {
      lhs.numVerts = rhs.numVerts;
      lhs.numEdges = rhs.numEdges;
      //flags
      lhs.flags = 0; //Have to start with a non-compound initialization
      if (rhs.isWeighted) { lhs.flags |= (CSR_header_flags.isWeighted : int(64)); }
      if (rhs.isZeroIndexed) { lhs.flags |= (CSR_header_flags.isZeroIndexed : int(64)); }
      if (rhs.isDirected) { lhs.flags |= (CSR_header_flags.isDirected : int(64)); }
      if (rhs.hasReverseEdges) { lhs.flags |= (CSR_header_flags.hasReverseEdges : int(64)); }
      if (rhs.isVertexT64) { lhs.flags |= (CSR_header_flags.isVertexT64 : int(64)); }
      if (rhs.isEdgeT64) { lhs.flags |= (CSR_header_flags.isEdgeT64 : int(64)); }
      if (rhs.isWeightT64) { lhs.flags |= (CSR_header_flags.isWeightT64 : int(64)); }
    }
    operator :(from : CSR_base, type to : this.type) {
      var tmp : to = from;
      return tmp;
    }
  }

  // New hierarchical concrete base that only holds descriptor vars
  class CSR_base {
    var numEdges : int(64) = 0;
    var numVerts : int(64) = 0;
    var isWeighted : bool = false;
    var isZeroIndexed : bool = false;
    var isDirected : bool = false;
    var hasReverseEdges : bool = false;
    var isVertexT64 : bool = false;
    var isEdgeT64 : bool = false;
    var isWeightT64 : bool = false;

    operator :(from : CSR_file_header, type to : this.type) {
      assert(from.binaryFormatVersion == CSR_BINARY_FORMAT_VERSION, "Assigning incompatible binary version ", from.binaryFormatVersion, " but expected ", CSR_BINARY_FORMAT_VERSION);
      var tmp = new to(
        numEdges = from.numEdges,
        numVerts = from.numVerts,
        isWeighted = if ((from.flags & (CSR_header_flags.isWeighted : int(64))) != 0) then true else false,
        isZeroIndexed = if ((from.flags & (CSR_header_flags.isZeroIndexed : int(64))) != 0) then true else false,
        isDirected = if ((from.flags & (CSR_header_flags.isDirected : int(64))) != 0) then true else false,
        hasReverseEdges = if ((from.flags & (CSR_header_flags.hasReverseEdges : int(64))) != 0) then true else false,
        isVertexT64 = if ((from.flags & (CSR_header_flags.isVertexT64 : int(64))) != 0) then true else false,
        isEdgeT64 = if ((from.flags & (CSR_header_flags.isEdgeT64 : int(64))) != 0) then true else false,
        isWeightT64 = if ((from.flags & (CSR_header_flags.isWeightT64 : int(64))) != 0) then true else false
       );
      return tmp;
    }
    override proc writeThis(f) throws {
      if (f.binary()) {
        //Construct a header from my descriptor and write it
        f.write(this : CSR_file_header);
      } else {
        var ret = "" : string;
        //concrete type and pointer and opening brace
        ret += stringify(this.type:string, ", ", this : c_void_ptr) + ": {";
        //Sizes
        ret += stringify("numEdges = ", numEdges, ", numVerts = ",  numVerts, ", ");
        //Flags
        ret += stringify("isWeighted = ", isWeighted, ", isVertexT64 = ", isVertexT64, ", isEdgeT64 = ",  isEdgeT64, ", isWeightT64 = ",  isWeightT64, ", isZeroIndexed = ", isZeroIndexed, ", isDirected = ",  isDirected, ", hasReverseEdges = ", hasReverseEdges);
        ret += "}";
        f.write(ret);
      }
    }
    //We can't mutate the actual type of the this instance to a CSR_arrays, so this will only ever assign header values. The client will have to use a bare base with MakeCSR/ReadCSRArrays itself
    override proc readThis(f) throws {
      if (f.binary()) {
        //Read the fixed-size header
        var header : CSR_file_header;
        f.read(header);
        //Convert the header to a CSR_base using overloaded cast
        var from = header : CSR_base;
        //Elementwise assign since we aren't allowed to overload class assignment, nor directly write to this
        this.numEdges = from.numEdges;
        this.numVerts = from.numVerts;
        this.isWeighted = from.isWeighted;
        this.isZeroIndexed = from.isZeroIndexed;
        this.isDirected = from.isDirected;
        this.hasReverseEdges = from.hasReverseEdges;
        this.isVertexT64 = from.isVertexT64;
        this.isEdgeT64 = from.isEdgeT64;
        this.isWeightT64 = from.isWeightT64;
      } else {
        assert(false, "CSR_base text read not supported!");
      }
    }
  }

  // New parameterized generic subclass that only holds graph arrays
  class CSR_arrays : CSR_base {
    //All arrays start with degenerate domains, and are modified at initialization
    param iWidth = 32; //either 32 or 64
    var idxDom : domain(1) = {0..0};
    var indices : [idxDom] int(iWidth);
    param oWidth = 32; //either 32 or 64
    var offDom : domain(1) = {0..0};
    var offsets : [offDom] int(oWidth);
    param wWidth = 32; //either 32 or 64
    var weightDom : domain(1) = {0..0};
    var weights : [weightDom] real(wWidth);

    override proc writeThis(f) throws {
      if (f.binary()) {
        super.writeThis(f);
        //Print offsets, then indices, then weights
        f.write(offsets);
        f.write(indices);
        if (isWeighted) { f.write(weights); }
      } else {
        var ret = "" : string;
        super.writeThis(f);
        ret += " -> ";
        ret += stringify(this.type:string, ", ", this : c_void_ptr) + ": {";
        //Emulate the default class writeThis, but with truncated array prints, and a pointer
        //Domains
        ret += stringify("idxDom = ", idxDom, ", offDom = ", offDom, ", weightDom = ", weightDom, ", ");
        //Truncated arrays
        ret += stringify("indices = [", indices[0..10], " ...], offsets = [", offsets[0..10], " ...], weights = [", weights[0..10], " ...]");
        //Closing brace
        ret += "}";
        f.write(ret);
      }
    }
    override proc readThis(f) throws{
      if (f.binary()) {
        //Assume we are at zero offset to re-read the header
        //Read the header and convert it to descriptor
        var header : CSR_file_header;
        f.read(header);
        //Convert the header to a base using operator overload
        var base = header : CSR_base;
        //Assert that all the fields match
	assert((this.isWeighted == base.isWeighted &&
                this.isZeroIndexed == base.isZeroIndexed &&
                this.isDirected == base.isDirected &&
                this.hasReverseEdges == base.hasReverseEdges &&
                this.isVertexT64 == base.isVertexT64 &&
                this.isEdgeT64 == base.isEdgeT64 &&
                this.isWeightT64 == base.isWeightT64 &&
                this.numEdges == base.numEdges &&
                this.numVerts == base.numVerts),
                "Error reading ", this.type : string, " from incompatible binary representation ", base : string);
        //Read arrays in order
        f.read(this.offsets);
        f.read(this.indices);
        if (isWeighted) { f.read(this.weights); }
      } else {
        assert(false, "CSR text read not supported!");
      }
    }
  }


proc NewCSRArrays(type CSR_type : CSR_arrays(?), in base : CSR_base): CSR_base {
  assert(( CSR_type.iWidth == (if base.isVertexT64 then 64 else 32) &&
           CSR_type.oWidth == (if base.isEdgeT64 then 64 else 32) &&
           CSR_type.wWidth == (if base.isWeightT64 then 64 else 32)
         ),
        "Cannot create new CSR_arrays, type mismatched with CSR_base!\nType: ", CSR_type : string, "\nCSR_base: ", base : string);
  var retCSR = new unmanaged CSR_type(
    numEdges = base.numEdges,
    numVerts = base.numVerts,
    isWeighted = base.isWeighted,
    isZeroIndexed = base.isZeroIndexed,
    isDirected = base.isDirected,
    hasReverseEdges = base.hasReverseEdges,
    isVertexT64 = base.isVertexT64,
    isEdgeT64 = base.isEdgeT64,
    isWeightT64 = base.isWeightT64,
    idxDom = {0..<base.numEdges},
    offDom = {0..base.numVerts},
    weightDom = {0..<(if base.isWeighted then base.numEdges else 0)}
  );
  return retCSR;
}

//This is what I'd like to be able to say
/*proc MakeCSR(in base : CSR_base) : CSR_base {
  return NewCSRArrays(CSR_arrays(iWidth=(if base.isVertexT64 then 64 else 32), oWidth=(if base.isEdgeT64 then 64 else 32), wWidth=(if base.isWeightT64 then 64 else 32)), base);
}*/
//This is how I currently have to say it
//This ladder lets us take the runtime booleans and translate them into a call
// to the right compile-time instantiation of the CSR type
private proc MakeCSR(in base : CSR_base, param iWidth : int, param oWidth : int) : CSR_base {
  return (if base.isWeightT64 then NewCSRArrays(CSR_arrays(iWidth, oWidth, 64), base) else NewCSRArrays(CSR_arrays(iWidth, oWidth, 32), base));
}
private proc MakeCSR(in base : CSR_base, param iWidth : int) : CSR_base {
  return (if base.isEdgeT64 then MakeCSR(base, iWidth, 64) else MakeCSR(base, iWidth, 32));
}
proc MakeCSR(in base : CSR_base) : CSR_base {
  return (if base.isVertexT64 then MakeCSR(base, 64) else MakeCSR(base, 32));
}

private proc ReadCSRArrays(in base : CSR_base, in channel, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) {
  var retArrays = try! (base : CSR_arrays(if isVertexT64 then 64 else 32, if isEdgeT64 then 64 else 32, if isWeightT64 then 64 else 32));
  channel.read(retArrays);
}
private proc ReadCSRArrays(in base : CSR_base, in channel, param isVertexT64 : bool, param isEdgeT64 : bool) {
  if (base.isWeightT64) {
    ReadCSRArrays(base, channel, isVertexT64, isEdgeT64, true);
  } else {
    ReadCSRArrays(base, channel, isVertexT64, isEdgeT64, false);
  }
}
private proc ReadCSRArrays(in base : CSR_base, in channel, param isVertexT64 : bool) {
  if (base.isEdgeT64) {
    ReadCSRArrays(base, channel, isVertexT64, true);
  } else {
    ReadCSRArrays(base, channel, isVertexT64, false);
  }
}
proc ReadCSRArrays(in base : CSR_base, in channel) {
  if (base.isVertexT64) {
    ReadCSRArrays(base, channel, true);
  } else {
    ReadCSRArrays(base, channel, false);
  }
}

proc readCSRFileToBase(in inFile : string) : CSR_base {
    ///File operations (which they are reworking as of 1.29.0)
    //FIXME: Add error handling
    //Open
    var readFile = IO.open(inFile, IO.iomode.r);
    //Create a read channel
    var readChannel = readFile.reader(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
    //Read the descriptor CSR_base
    var desc = new unmanaged CSR_base();
    readChannel.read(desc);
    //Make the actual CSR_arrays based on the descriptor base
    var retArrays = MakeCSR(desc);
    //Rewind the file cursor to zero offset, with unbounded range
    readChannel.seek(0..);
    //Invoke the param-spec ladder to read the actual CSR member
    ReadCSRArrays(retArrays, readChannel);
    //TODO anything to gracefully close the channel/file?
    return retArrays;
}

proc writeCSRFile(in outFile : string, in base : CSR_base) {
  //Open the file
  var writeFile = IO.open(outFile, IO.iomode.cw);
  //Create a write channel
  var writeChannel = writeFile.writer(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
  //Write the data arrays
  writeChannel.write(base);
  //TODO anything to gracefully close the channel/file?
}
}
