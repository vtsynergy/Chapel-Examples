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

  record CSR_file_header {
    var binaryFormatVersion : int(64) = 1;
    var numVerts : int(64) = 0;
    var numEdges : int(64) = 0;
    //Because of how Chapel casts to enums we can't store "all false (0)" or ORed values in an enum, so the flags field has to be treated as int(64)
    var flags : int(64) = 0;
  }

  //Runtime type descriptor
  record CSR_descriptor {
    var isWeighted : bool;
    var isVertexT64 : bool;
    var isEdgeT64 : bool;
    var isWeightT64 : bool;
    var numEdges : int(64);
    var numVerts : int(64);
   proc init=(rhs : (4*bool, 2*int)) {
      this.isWeighted = rhs(0);
      this.isVertexT64 = rhs(1);
      this.isEdgeT64 = rhs(2);
      this.isWeightT64 = rhs(3);
      this.numEdges = rhs(4);
      this.numVerts = rhs(5);
    }
   proc init=(rhs : CSR_descriptor) {
      this.isWeighted = rhs.isWeighted;
      this.isVertexT64 = rhs.isVertexT64;
      this.isEdgeT64 = rhs.isEdgeT64;
      this.isWeightT64 = rhs.isWeightT64;
      this.numEdges = rhs.numEdges;
      this.numVerts = rhs.numVerts;
    }
    operator =(ref lhs : CSR_descriptor, rhs : CSR_descriptor) {
      lhs.isWeighted = rhs.isWeighted;
      lhs.isVertexT64 = rhs.isVertexT64;
      lhs.isEdgeT64 = rhs.isEdgeT64;
      lhs.isWeightT64 = rhs.isWeightT64;
      lhs.numEdges = rhs.numEdges;
      lhs.numVerts = rhs.numVerts;
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
      //Read the fixed-size header
      //  var header = {0, 0, 0, 0} : CSR_file_header; // "illegal cast from DefaultAssociativeDom(int(64),true) to CSR_file_header" // so I guess don't initialize here?
      var header : CSR_file_header;
      var expectedBinFmt = header.binaryFormatVersion; //FIXME I can't figure out a better way to grab the default integral constant from the record type, other than just copying it from an entity that has been default initialized
      f.read(header);
      var actualBinFmt : int(64);
      var numVerts : int(64);
      var numEdges : int(64);
      var isWeighted : bool;
      var isZeroIndexed : bool;
      var isDirected : bool;
      var hasReverseEdges : bool;
      var isVertexT64 : bool;
      var isEdgeT64 : bool;
      var isWeightT64 : bool;
      //FIXME This is a CSR_descriptor, so we've got it, and the one in the eventual handle, we should really only have one
      var myCSR = parseCSRHeader(header, actualBinFmt, numVerts, numEdges, isWeighted, isZeroIndexed, isDirected, hasReverseEdges, isVertexT64, isEdgeT64, isWeightT64);
      //Assert that the binary format version is the one we're expecting (Vers. 2)
      assert(actualBinFmt == expectedBinFmt, "Binary version of inFile is ", header.binaryFormatVersion, " but expected ", expectedBinFmt);
      this = MakeCSR(myCSR.isWeighted, myCSR.isVertexT64, myCSR.isEdgeT64, myCSR.isWeightT64, numEdges, numVerts);
      ReadCSRArrays(this, f, isZeroIndexed, isDirected, hasReverseEdges);
    }
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

    //TODO implement readThis, which is harder because we don't know the concrete type until we've read the header.
    //One approach would be to implement readThis to also read the header, and then if type widths of the header don't match, then coerce them into whatever width the instance is configured for. For our application, if we wanted to use the actual type, we'd have to read the header, create the instance, and then rewind the seek cursor to before the header. Which is fine. But this would then support implicit read-time conversion which could be handy for promoting/demoting widths.

    //writeThis is easier to implement because we already know the concrete type
    override proc writeThis(f) throws {
      if (f.binary()) { //We assume binary IO is for file writing and non-binary is for string
        //Construct a header and write it
        var header : CSR_file_header;
        header.numVerts = numVerts;
        header.numEdges = numEdges;
        //flags
        if (isWeighted) { header.flags |= (CSR_header_flags.isWeighted : int(64)); }
        if (isZeroIndexed) { header.flags |= (CSR_header_flags.isZeroIndexed : int(64)); }
        if (isDirected) { header.flags |= (CSR_header_flags.isDirected : int(64)); }
        if (hasReverseEdges) { header.flags |= (CSR_header_flags.hasReverseEdges : int(64)); }
        if (isVertexT64) { header.flags |= (CSR_header_flags.isVertexT64 : int(64)); }
        if (isEdgeT64) { header.flags |= (CSR_header_flags.isEdgeT64 : int(64)); }
        if (isWeightT64) { header.flags |= (CSR_header_flags.isWeightT64 : int(64)); }
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
  }


proc NewCSRHandle(type CSR_type : CSR(?), in numEdges : int(64), in numVerts : int(64)): CSR_handle {
  var retHandle : CSR_handle;
  local { // Right now the GPU implementation uses "wide" pointers everywhere, "local" forces a version that doesn't trip up on node-locality assertions for now
    var retCSR = new unmanaged CSR_type(numEdges, numVerts);
    retCSR.numEdges = numEdges;
    retCSR.numVerts = numVerts;
    var retCast = (retCSR : c_void_ptr); //In 2.0 this *may* become analagous to c_ptrTo(<someclass>) but it isn't yet
    retHandle.desc = new CSR_descriptor(CSR_type.isWeighted, CSR_type.isVertexT64, CSR_type.isEdgeT64, CSR_type.isWeightT64, numEdges, numVerts);
    retHandle.data = retCast;
  }

  return retHandle;
}

//This ladder lets us take the runtime booleans and translate them into a call
// to the right compile-time instantiation of the CSR type
//We then pass the opaque handle up to be passed around by the functions that
// don't really need to know the internals of the type
proc MakeCSR(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool, in numEdges : int(64), in numVerts : int(64)) :CSR_handle {
  return NewCSRHandle(CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), numEdges, numVerts);
}
proc MakeCSR(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, in isWeightT64 : bool, in numEdges : int(64), in numVerts : int(64)) :CSR_handle {
  if (isWeightT64) {
    return MakeCSR(isWeighted, isVertexT64, isEdgeT64, true, numEdges, numVerts);
  } else {
    return MakeCSR(isWeighted, isVertexT64, isEdgeT64, false, numEdges, numVerts);
  }
} 
proc MakeCSR(param isWeighted : bool, param isVertexT64 : bool, in isEdgeT64 : bool, in isWeightT64 : bool, in numEdges : int(64), in numVerts : int(64)) :CSR_handle {
  if (isEdgeT64) {
    return MakeCSR(isWeighted, isVertexT64, true, isWeightT64, numEdges, numVerts);
  } else {
    return MakeCSR(isWeighted, isVertexT64, false, isWeightT64, numEdges, numVerts);
  }
}
proc MakeCSR(param isWeighted : bool, in isVertexT64 : bool, in isEdgeT64 : bool, in isWeightT64 : bool, in numEdges : int(64), in numVerts : int(64)) :CSR_handle {
  if (isVertexT64) {
    return MakeCSR(isWeighted, true, isEdgeT64, isWeightT64, numEdges, numVerts);
  } else {
    return MakeCSR(isWeighted, false, isEdgeT64, isWeightT64, numEdges, numVerts);
  }
}
proc MakeCSR(in isWeighted : bool, in isVertexT64 : bool, in isEdgeT64 : bool, in isWeightT64 : bool, in numEdges = 0 : int(64), in numVerts = 0 : int(64)) :CSR_handle {
  if (isWeighted) {
    return MakeCSR(true, isVertexT64, isEdgeT64, isWeightT64, numEdges, numVerts);
  } else {
    return MakeCSR(false, isVertexT64, isEdgeT64, isWeightT64, numEdges, numVerts);
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

proc ReadCSRArrays(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool, in handle : CSR_handle, in channel, in isZeroIndexed: bool, in isDirected : bool, in hasReverseEdges : bool) {
  //Bring the handle into concrete type
  var myCSR = ReinterpretCSRHandle(unmanaged CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), handle);
  //FIXME: These should be read in differently
  myCSR.isZeroIndexed = isZeroIndexed;
  myCSR.isDirected = isDirected;
  myCSR.hasReverseEdges = hasReverseEdges;
  //Read arrays 
  channel.read(myCSR.offsets);
  channel.read(myCSR.indices);
  if (isWeighted) { channel.read(myCSR.weights); }
  writeln("After array read: ", myCSR);
}
proc ReadCSRArrays(param isWeighted : bool, param isVertexT64 : bool, param isEdgeT64 : bool, in handle : CSR_handle, in channel, in isZeroIndexed: bool, in isDirected : bool, in hasReverseEdges : bool) {
  if (handle.desc.isWeightT64) {
    ReadCSRArrays(isWeighted, isVertexT64, isEdgeT64, true, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  } else {
    ReadCSRArrays(isWeighted, isVertexT64, isEdgeT64, false, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  }
} 
proc ReadCSRArrays(param isWeighted : bool, param isVertexT64 : bool, in handle : CSR_handle, in channel, in isZeroIndexed: bool, in isDirected : bool, in hasReverseEdges : bool) {
  if (handle.desc.isEdgeT64) {
    ReadCSRArrays(isWeighted, isVertexT64, true, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  } else {
    ReadCSRArrays(isWeighted, isVertexT64, false, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  }
}
proc ReadCSRArrays(param isWeighted : bool, in handle : CSR_handle, in channel, in isZeroIndexed: bool, in isDirected : bool, in hasReverseEdges : bool) {
  if (handle.desc.isVertexT64) {
    ReadCSRArrays(isWeighted, true, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  } else {
    ReadCSRArrays(isWeighted, false, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  }
}
proc ReadCSRArrays(in handle : CSR_handle, in channel, in isZeroIndexed: bool, in isDirected : bool, in hasReverseEdges : bool) {
  if (handle.desc.isWeighted) {
    ReadCSRArrays(true, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  } else {
    ReadCSRArrays(false, handle, channel, isZeroIndexed, isDirected, hasReverseEdges);
  }
} 

//The new parser returns the elaborated CSR type, so that the application can use it directly to construct
proc parseCSRHeader(in header : CSR_file_header,out binFmtVers : int(64), out numVerts : int(64), out numEdges : int(64),
    out isWeighted : bool, out isZeroIndexed : bool, out isDirected : bool, out hasReverseEdges : bool,
    out isVertexT64 : bool, out isEdgeT64: bool, ref isWeightT64: bool) : CSR_descriptor {
  binFmtVers = header.binaryFormatVersion;
  numVerts = header.numVerts;
  numEdges = header.numEdges; 
  if ((header.flags & (CSR_header_flags.isWeighted : int(64))) != 0) { isWeighted = true; }
  if ((header.flags & (CSR_header_flags.isZeroIndexed : int(64))) != 0) { isZeroIndexed = true; }
  if ((header.flags & (CSR_header_flags.isDirected : int(64))) != 0) { isDirected = true; }
  if ((header.flags & (CSR_header_flags.hasReverseEdges : int(64))) != 0) { hasReverseEdges = true; }
  if ((header.flags & (CSR_header_flags.isVertexT64 : int(64))) != 0) { isVertexT64 = true; }
  if ((header.flags & (CSR_header_flags.isEdgeT64 : int(64))) != 0) { isEdgeT64 = true; }
  if ((header.flags & (CSR_header_flags.isWeightT64 : int(64))) != 0) { isWeightT64 = true; }
  return new CSR_descriptor(isWeighted, isVertexT64, isEdgeT64, isWeightT64, numVerts, numEdges);
}

class food {
  var bar : int(64);
};


//FIXME, remove these three bools one the writer is encapsulated in CSR.writeThis
proc readCSRFile(in inFile : string, out isZeroIndexed : bool, out isDirected : bool, out hasReverseEdges : bool) : CSR_handle {

    ///File operations (which they are reworking as of 1.29.0)
    //FIXME: Add error handling
    //FIXME: Reimplement using readThis methods
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
