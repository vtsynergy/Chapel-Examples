//Has to be prototype because the IO calls throw. Apparently
prototype module CSR {
  use IO; //Need for file ops
  use CTypes; //Need for handle's void pointer
  //Need to implement a record for the CSRv2 header format
  // This might be a pain because it doesn't look like Chapel has bitfields, so we will need to create a bitmask enum to make operations simpler

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
    proc init=(rhs : CSR_descriptor) {
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
    operator =(ref lhs: CSR_file_header, rhs : CSR_descriptor) {
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
    operator :(from : CSR_descriptor, type to : this.type) {
      var tmp : to = from;
      return tmp;
    }
  }

  //Runtime type descriptor
  record CSR_descriptor {
    var isWeighted : bool = false;
    var isZeroIndexed : bool = false;
    var isDirected : bool = false;
    var hasReverseEdges : bool = false;
    var isVertexT64 : bool = false;
    var isEdgeT64 : bool = false;
    var isWeightT64 : bool = false;
    var numEdges : int(64) = 0;
    var numVerts : int(64) = 0;
    //need a general init function now, but it doesn't have to do anything since all fields have defaults
    proc init() { }
    proc init=(rhs : CSR_descriptor) {
      this.isWeighted = rhs.isWeighted;
      this.isZeroIndexed = rhs.isZeroIndexed;
      this.isDirected = rhs.isDirected;
      this.hasReverseEdges = rhs.hasReverseEdges;
      this.isVertexT64 = rhs.isVertexT64;
      this.isEdgeT64 = rhs.isEdgeT64;
      this.isWeightT64 = rhs.isWeightT64;
      this.numEdges = rhs.numEdges;
      this.numVerts = rhs.numVerts;
    }
    operator =(ref lhs: CSR_descriptor, rhs : CSR_descriptor) {
      lhs.isWeighted = rhs.isWeighted;
      lhs.isZeroIndexed = rhs.isZeroIndexed;
      lhs.isDirected = rhs.isDirected;
      lhs.hasReverseEdges = rhs.hasReverseEdges;
      lhs.isVertexT64 = rhs.isVertexT64;
      lhs.isEdgeT64 = rhs.isEdgeT64;
      lhs.isWeightT64 = rhs.isWeightT64;
      lhs.numEdges = rhs.numEdges;
      lhs.numVerts = rhs.numVerts;
    }
    proc init=(rhs : CSR_file_header) {
      assert(rhs.binaryFormatVersion == CSR_BINARY_FORMAT_VERSION, "Assigning incompatible binary version ", rhs.binaryFormatVersion, " but expected ", CSR_BINARY_FORMAT_VERSION);
      if ((rhs.flags & (CSR_header_flags.isWeighted : int(64))) != 0) { this.isWeighted = true; }
      if ((rhs.flags & (CSR_header_flags.isZeroIndexed : int(64))) != 0) { this.isZeroIndexed = true; }
      if ((rhs.flags & (CSR_header_flags.isDirected : int(64))) != 0) { this.isDirected = true; }
      if ((rhs.flags & (CSR_header_flags.hasReverseEdges : int(64))) != 0) { this.hasReverseEdges = true; }
      if ((rhs.flags & (CSR_header_flags.isVertexT64 : int(64))) != 0) { this.isVertexT64 = true; }
      if ((rhs.flags & (CSR_header_flags.isEdgeT64 : int(64))) != 0) { this.isEdgeT64 = true; }
      if ((rhs.flags & (CSR_header_flags.isWeightT64 : int(64))) != 0) { this.isWeightT64 = true; }
      this.numEdges = rhs.numEdges;
      this.numVerts = rhs.numVerts;
    }
    operator =(ref lhs: CSR_descriptor, rhs : CSR_file_header) {
      assert(rhs.binaryFormatVersion == CSR_BINARY_FORMAT_VERSION, "Assigning incompatible binary version ", rhs.binaryFormatVersion, " but expected ", CSR_BINARY_FORMAT_VERSION);
      if ((rhs.flags & (CSR_header_flags.isWeighted : int(64))) != 0) { lhs.isWeighted = true; }
      if ((rhs.flags & (CSR_header_flags.isZeroIndexed : int(64))) != 0) { lhs.isZeroIndexed = true; }
      if ((rhs.flags & (CSR_header_flags.isDirected : int(64))) != 0) { lhs.isDirected = true; }
      if ((rhs.flags & (CSR_header_flags.hasReverseEdges : int(64))) != 0) { lhs.hasReverseEdges = true; }
      if ((rhs.flags & (CSR_header_flags.isVertexT64 : int(64))) != 0) { lhs.isVertexT64 = true; }
      if ((rhs.flags & (CSR_header_flags.isEdgeT64 : int(64))) != 0) { lhs.isEdgeT64 = true; }
      if ((rhs.flags & (CSR_header_flags.isWeightT64 : int(64))) != 0) { lhs.isWeightT64 = true; }
      lhs.numEdges = rhs.numEdges;
      lhs.numVerts = rhs.numVerts;
    }
    operator :(from : CSR_file_header, type to : this.type) {
      var tmp : to = from;
      return tmp;
    }
  }
    //FIXME: These really belong to the CSR_handle record, but private cannot be applied to members yet
    private proc writeCSRHandle(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool, in handle : CSR_handle, in channel) {
      //Bring the handle into concrete type
      var myCSR = ReinterpretCSRHandle(unmanaged CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), handle);
      //Then write the concrete instance
      channel.write(myCSR);
    }
    private proc writeCSRHandle(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, in handle : CSR_handle, in channel) {
      if (handle.desc.isWeightT64) {
        writeCSRHandle(isWeighted, isVertexT64, isEdgeT64, true, handle, channel);
      } else {
        writeCSRHandle(isWeighted, isVertexT64, isEdgeT64, false, handle, channel);
      }
    }
    private proc writeCSRHandle(param isWeighted : bool, param isVertexT64 : bool, in handle : CSR_handle, in channel) {
      if (handle.desc.isEdgeT64) {
        writeCSRHandle(isWeighted, isVertexT64, true, handle, channel);
      } else {
        writeCSRHandle(isWeighted, isVertexT64, false, handle, channel);
      }
    }
    private proc writeCSRHandle(param isWeighted : bool, in handle : CSR_handle, in channel) {
      if (handle.desc.isVertexT64) {
        writeCSRHandle(isWeighted, true, handle, channel);
      } else {
        writeCSRHandle(isWeighted, false, handle, channel);
      }
    }
    private proc writeCSRHandle(in handle : CSR_handle, in channel) {
      if (handle.desc.isWeighted) {
        writeCSRHandle(true, handle, channel);
      } else {
        writeCSRHandle(false, handle, channel);
      }
    }
  //Opaque handle
  record CSR_handle {
    var desc : CSR_descriptor;
    var data : c_void_ptr;
    //only class methods override I guess
    proc writeThis(f) throws {
      if (f.binary()) { //We assume binary IO is for file writing and non-binary is for string
        if (data != nil) {
          writeCSRHandle(this, f);
        }
      } else {
        //descriptor
        f.write("(desc = ", desc, ", ");
        //pointer
        f.write("data = ");
        //Check for valid data
        if (data != nil) {
          writeCSRHandle(this, f);
        } else {
          f.write(data);
        }
        //closing paren
        f.write(")");
      }
    }
    proc readThis(f) throws {
      if (f.binary()) {
        //Read the fixed-size header
        var header : CSR_file_header;
        f.read(header);
        //Convert the header to a descriptor using record operator overload
        this.desc = header;
        this = MakeCSR(this.desc);
        //Rewind the file cursor to zero offset, with unbounded range
        f.seek(0..);
        //Invoke the param-spec ladder to read the actual CSR member
        ReadCSRArrays(this, f);
      } else {
        assert(false, "CSR_handle text read not supported!");
      }
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
  }

  //Can we make this a generic type to accept both 32- and 64-bit vertices/edges/weights?
  class CSR {
    //TODO Atharva wanted me to confirm this structure can deal with directed graphs for TC, so that we don't have as much redundant work
    var numEdges : int(64);
    var numVerts : int(64);
    param isWeighted : bool;
    var isZeroIndexed : bool;
    var isDirected : bool;
    var hasReverseEdges : bool;
    param isVertexT64 : bool;
    param isEdgeT64 : bool;
    param isWeightT64 : bool;
    var idxDom : domain(1) = {0..numEdges-1};
    var indices : [idxDom] int(if isVertexT64 then 64 else 32);
    var offDom : domain(1) = {0..(numVerts)};
    var offsets : [offDom] int(if isEdgeT64 then 64 else 32);
    var weightDom : domain(1) = {0..(if isWeighted then numEdges-1 else 0)}; //Degenerate if we don't have weights
    var weights : [weightDom] real(if isWeightT64 then 64 else 32);

    proc getDescriptor() : CSR_descriptor {
      var ret : CSR_descriptor;
      ret.isWeighted = this.isWeighted;
      ret.isZeroIndexed = this.isZeroIndexed;
      ret.isDirected = this.isDirected;
      ret.hasReverseEdges = this.hasReverseEdges;
      ret.isVertexT64 = this.isVertexT64;
      ret.isEdgeT64 = this.isEdgeT64;
      ret.isWeightT64 = this.isWeightT64;
      ret.numEdges = this.numEdges;
      ret.numVerts = this.numVerts;
      return ret;
    }
    //writeThis is easier to implement because we already know the concrete type
    override proc writeThis(f) throws {
      if (f.binary()) { //We assume binary IO is for file writing and non-binary is for string
        //Construct a header from my descriptor and write it
        var header = this.getDescriptor() : CSR_file_header;
        f.write(header);
        
        //Print offsets, then indices, then weights
        f.write(offsets);
        f.write(indices);
        if (isWeighted) { f.write(weights); }
      } else {
        //Emulate the default class writeThis, but with truncated array prints, and a pointer
        var ret = "" : string;
        //concrete type and pointer and opening brace
        ret += stringify(this.type:string, ", ", this : c_void_ptr) + ": {";
        //Sizes
        ret += stringify("numEdges = ", numEdges, ", numVerts = ",  numVerts, ", ");
        //Flags
        ret += stringify("isWeighted = ", isWeighted, ", isVertexT64 = ", isVertexT64, ", isEdgeT64 = ",  isEdgeT64, ", isWeightT64 = ",  isWeightT64, ", isZeroIndexed = ", isZeroIndexed, ", isDirected = ",  isDirected, ", hasReverseEdges = ", hasReverseEdges, ", ");
        //Domains
        ret += stringify("idxDom = ", idxDom, ", offDom = ", offDom, ", weightDom = ", weightDom, ", ");
        //Truncated arrays
	ret += stringify("indices = [", indices[0..10], " ...], offsets = [", offsets[0..10], " ...], weights = [", weights[0..10], " ...]");
	//Closing brace
        ret += "}";
        f.write(ret); 
      }
    }
    override proc readThis(f) {
      if (f.binary()) {
        //Assume we are at zero offset to re-read the header
        //Read the header and convert it to descriptor
        var header : CSR_file_header;
        f.read(header);
        //Convert the header to a descriptor using record operator overload
        var desc : CSR_descriptor = header;
        //Assert that all the fields match
	assert((this.isWeighted == desc.isWeighted &&
                this.isZeroIndexed == desc.isZeroIndexed &&
                this.isDirected == desc.isDirected &&
                this.hasReverseEdges == desc.hasReverseEdges &&
                this.isVertexT64 == desc.isVertexT64 &&
                this.isEdgeT64 == desc.isEdgeT64 &&
                this.isWeightT64 == desc.isWeightT64 &&
                this.numEdges == desc.numEdges &&
                this.numVerts == this.numVerts),
                "Error reading ", this.type : string, " from incompatible binary representation ", desc : string);
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
  var retCSR = new unmanaged CSR_type(idxDom = {0..<base.numEdges}, offDom = {0..base.numVerts}, weightDom = {0..<(if base.isWeighted then base.numEdges else 0)});
  return retCSR;
}

proc NewCSRHandle(type CSR_type : CSR(?), in desc : CSR_descriptor): CSR_handle {
  assert(( CSR_type.isWeighted == desc.isWeighted &&
           CSR_type.isVertexT64 == desc.isVertexT64 &&
           CSR_type.isEdgeT64 == desc.isEdgeT64 &&
           CSR_type.isWeightT64 == desc.isWeightT64),
           "Cannot create new CSR handle, type mismatched with descriptor!\nType: ", CSR_type : string, "\nDescriptor: ", desc : string);
  var retHandle : CSR_handle;
  local { // Right now the GPU implementation uses "wide" pointers everywhere, "local" forces a version that doesn't trip up on node-locality assertions for now
    //FIXME add an "initialize from descriptor" procedure
    var retCSR = new unmanaged CSR_type(desc.numEdges, desc.numVerts);
    //Assign all the non-param, non-array fields
    retCSR.isZeroIndexed = desc.isZeroIndexed;
    retCSR.isDirected = desc.isDirected;
    retCSR.hasReverseEdges = desc.hasReverseEdges;
    retCSR.numEdges = desc.numEdges;
    retCSR.numVerts = desc.numVerts;
    var retCast = (retCSR : c_void_ptr); //In 2.0 this *may* become analagous to c_ptrTo(<someclass>) but it isn't yet
    retHandle.data = retCast;
    retHandle.desc = desc;
  }
  return retHandle;
}

//This ladder lets us take the runtime booleans and translate them into a call
// to the right compile-time instantiation of the CSR type
//We then pass the opaque handle up to be passed around by the functions that
// don't really need to know the internals of the type
private proc MakeCSR(in desc : CSR_descriptor, param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) :CSR_handle {
  return NewCSRHandle(CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), desc);
}
private proc MakeCSR(in desc : CSR_descriptor, param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool) :CSR_handle {
  if (desc.isWeightT64) {
    return MakeCSR(desc, isWeighted, isVertexT64, isEdgeT64, true);
  } else {
    return MakeCSR(desc, isWeighted, isVertexT64, isEdgeT64, false);
  }
} 
private proc MakeCSR(in desc : CSR_descriptor, param isWeighted : bool, param isVertexT64 : bool) :CSR_handle {
  if (desc.isEdgeT64) {
    return MakeCSR(desc, isWeighted, isVertexT64, true);
  } else {
    return MakeCSR(desc, isWeighted, isVertexT64, false);
  }
}
private proc MakeCSR(in desc: CSR_descriptor, param isWeighted : bool) :CSR_handle {
  if (desc.isVertexT64) {
    return MakeCSR(desc, isWeighted, true);
  } else {
    return MakeCSR(desc, isWeighted, false);
  }
}
proc MakeCSR(in desc : CSR_descriptor) :CSR_handle {
  if (desc.isWeighted) {
    return MakeCSR(desc, true);
  } else {
    return MakeCSR(desc, false);
  }
} 

proc ReinterpretCSRHandle(type CSR_type: unmanaged CSR(?), in handle : CSR_handle) : CSR_type {
  var retCSR : CSR_type;

  local {
    assert(handle.desc.isWeighted == CSR_type.isWeighted &&
      handle.desc.isVertexT64 == CSR_type.isVertexT64 &&
      handle.desc.isEdgeT64 == CSR_type.isEdgeT64 &&
      handle.desc.isWeightT64 == CSR_type.isWeightT64,
      //This can only print the descriptor member, or we risk infinite stack recursion when handle.writeThis calls the CSR.writeThis on a non-nil data member
      "Provided CSR_handle: ", handle.desc : string, " incompatible with reinterpreted type: ", CSR_type : string);

    //Open the handle
    retCSR = ((handle.data : CSR_type?) : CSR_type); //Have to cast twice here, not allowed to directly go from c_void_ptr to non-nillable class, because it eliminates the chance for a runtime check of nil value
  }
  return retCSR;
}

private proc ReadCSRArrays(in handle : CSR_handle, in channel, param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) {
  //Bring the handle into concrete type
  var myCSR = ReinterpretCSRHandle(unmanaged CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), handle);
  //Read arrays 
  channel.read(myCSR);
}
private proc ReadCSRArrays(in handle : CSR_handle, in channel, param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool) {
  if (handle.desc.isWeightT64) {
    ReadCSRArrays(handle, channel, isWeighted, isVertexT64, isEdgeT64, true);
  } else {
    ReadCSRArrays(handle, channel, isWeighted, isVertexT64, isEdgeT64, false);
  }
} 
private proc ReadCSRArrays(in handle : CSR_handle, in channel, param isWeighted : bool, param isVertexT64 : bool) {
  if (handle.desc.isEdgeT64) {
    ReadCSRArrays(handle, channel, isWeighted, isVertexT64, true);
  } else {
    ReadCSRArrays(handle, channel, isWeighted, isVertexT64, false);
  }
}
private proc ReadCSRArrays(in handle : CSR_handle, in channel, param isWeighted : bool) {
  if (handle.desc.isVertexT64) {
    ReadCSRArrays(handle, channel, isWeighted, true);
  } else {
    ReadCSRArrays(handle, channel, isWeighted, false);
  }
}
proc ReadCSRArrays(in handle : CSR_handle, in channel) {
  if (handle.desc.isWeighted) {
    ReadCSRArrays(handle, channel, true);
  } else {
    ReadCSRArrays(handle, channel, false);
  }
} 

proc readCSRFile(in inFile : string) : CSR_handle {

    ///File operations (which they are reworking as of 1.29.0)
    //FIXME: Add error handling
    //Open
    var readFile = IO.open(inFile, IO.iomode.r);
    //Create a read channel
    var readChannel = readFile.reader(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
    //Create an empty handle
    var retHandle : CSR_handle;
    readChannel.read(retHandle);
    //TODO anything to gracefully close the channel/file?
    return retHandle;
}

proc writeCSRFile(in outFile : string, in handle : CSR_handle) {
  //Open the file
  var writeFile = IO.open(outFile, IO.iomode.cw);
  //Create a write channel
  var writeChannel = writeFile.writer(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
  //Write the data arrays
  writeChannel.write(handle);
  //TODO anything to gracefully close the channel/file?
}

}
