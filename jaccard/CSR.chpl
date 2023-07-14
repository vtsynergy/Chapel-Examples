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
    operator :(from : CSR_base, type to : this.type) {
      var tmp = new to();
      tmp.numEdges = from.numEdges;
      tmp.numVerts = from.numVerts;
      tmp.isWeighted = from.isWeighted;
      tmp.isZeroIndexed = from.isZeroIndexed;
      tmp.isDirected = from.isDirected;
      tmp.hasReverseEdges = from.hasReverseEdges;
      tmp.isVertexT64 = from.isVertexT64;
      tmp.isEdgeT64 = from.isEdgeT64;
      tmp.isWeightT64 = from.isWeightT64;
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

    operator :(from : CSR_descriptor, type to : this.type) {
      return new to(
        numEdges = from.numEdges,
        numVerts = from.numVerts,
        isWeighted = from.isWeighted,
        isZeroIndexed = from.isZeroIndexed,
        isDirected = from.isDirected,
        hasReverseEdges = from.hasReverseEdges,
        isVertexT64 = from.isVertexT64,
        isEdgeT64 = from.isEdgeT64,
        isWeightT64 = from.isWeightT64
      );
    }
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

    operator :(from : ?fromType, type to : this.type) where isSubtype(fromType, CSR(?)) {
      assert((to.iWidth == (if fromType.isVertexT64 then 64 else 32) &&
              to.oWidth == (if fromType.isEdgeT64 then 64 else 32) &&
              to.wWidth == (if fromType.isWeightT64 then 64 else 32)),
             "Cannot cast between!\nfromType: ", fromType : string, "\ntoType: ", to : string);
      var tmp = new to(
        numEdges = from.numEdges,
        numVerts = from.numVerts,
        isWeighted = from.isWeighted,
        isZeroIndexed = from.isZeroIndexed,
        isDirected = from.isDirected,
        hasReverseEdges = from.hasReverseEdges,
        isVertexT64 = from.isVertexT64,
        isEdgeT64 = from.isEdgeT64,
        isWeightT64 = from.isWeightT64,
        idxDom = from.idxDom,
        indices = from.indices,
        offDom = from.offDom,
        offsets = from.offsets,
        weightDom = from.weightDom,
        weights = from.weights
      );
      return tmp;
    }
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

    operator :(from : ?fromType, type to : this.type) where isSubtype(fromType, CSR_arrays(?)) {
      assert((to.isVertexT64 == (fromType.iWidth == 64) &&
              to.isEdgeT64 == (fromType.oWidth == 64) &&
              to.isWeightT64 == (fromType.wWidth == 64)),
             "Cannot cast between!\nfromType: ", fromType : string, "\ntoType: ", to : string);
      var tmp = new to(
        numEdges = from.numEdges,
        numVerts = from.numVerts,
        isZeroIndexed = from.isZeroIndexed,
        isDirected = from.isDirected,
        hasReverseEdges = from.hasReverseEdges,
        idxDom = from.idxDom,
        indices = from.indices,
        offDom = from.offDom,
        offsets = from.offsets,
        weightDom = from.weightDom,
        weights = from.weights
      );
      return tmp;
    }

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
  writeln("Created new CSR_arrays: ", retCSR);
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
    type retType = CSR_arrays((if CSR_type.isVertexT64 then 64 else 32), (if CSR_type.isEdgeT64 then 64 else 32), (if CSR_type.isWeightT64 then 64 else 32));
    var retArrays = (NewCSRArrays(retType, (desc : CSR_base)) : retType);
    var retCSR = (retArrays : unmanaged CSR_type);
    delete retArrays;
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
proc readCSRFileToHandle(in inFile : string) : CSR_handle {
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

proc writeCSRFile(in outFile : string, in base : CSR_base) {
  //Open the file
  var writeFile = IO.open(outFile, IO.iomode.cw);
  //Create a write channel
  var writeChannel = writeFile.writer(kind = IO.iokind.native, locking = false, hints = IO.ioHintSet.sequential);
  writeln(base);
  //Write the data arrays
  writeChannel.write(base);
  //TODO anything to gracefully close the channel/file?
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

private proc deepCastToHandle(in base : CSR_base, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool, param isWeighted : bool) : CSR_handle {
  type retType = CSR(isVertexT64 = isVertexT64, isEdgeT64 = isEdgeT64, isWeightT64 = isWeightT64, isWeighted = isWeighted);
  var retCSR : CSR_handle;
  local {
    retCSR = MakeCSR(base : CSR_descriptor);
    var fullCSR = ReinterpretCSRHandle(unmanaged retType, retCSR);
    fullCSR = ((base : CSR_arrays(if isVertexT64 then 64 else 32, if isEdgeT64 then 64 else 32, if isWeightT64 then 64 else 32)) : retType);
    retCSR.data = (fullCSR : c_void_ptr);
    writeln("Full CSR: ", fullCSR);
  }
  return retCSR;
}
private proc deepCastToHandle(in base : CSR_base, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) : CSR_handle {
  return (if (base.isWeighted) then deepCastToHandle(base, isVertexT64, isEdgeT64, isWeightT64, true) else deepCastToHandle(base, isVertexT64, isEdgeT64, isWeightT64, false));
}
private proc deepCastToHandle(in base : CSR_base, param isVertexT64 : bool, param isEdgeT64 : bool) : CSR_handle {
  return (if (base.isWeightT64) then deepCastToHandle(base, isVertexT64, isEdgeT64, true) else deepCastToHandle(base, isVertexT64, isEdgeT64, false));
}
private proc deepCastToHandle(in base : CSR_base, param isVertexT64 : bool) : CSR_handle {
  return (if (base.isEdgeT64) then deepCastToHandle(base, isVertexT64, true) else deepCastToHandle(base, isVertexT64, false));
}
//THIS WILL CREATE A COPY
proc deepCastToHandle(in base : CSR_base) : CSR_handle {
  return (if (base.isVertexT64) then deepCastToHandle(base, true) else deepCastToHandle(base, false));
}

private proc deepCastToBase(in handle : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool, param isWeighted : bool) : CSR_base {
  type retType = CSR_arrays((if isVertexT64 then 64 else 32), (if isEdgeT64 then 64 else 32), (if isWeightT64 then 64 else 32));
  var retArrays = (NewCSRArrays(retType, (handle.desc : CSR_base)) : retType);
  local { // Right now the GPU implementation uses "wide" pointers everywhere, "local" forces a version that doesn't trip up on node-locality assertions for now
    var copyCSR = ReinterpretCSRHandle(unmanaged CSR(isWeighted, isVertexT64, isEdgeT64, isWeightT64), handle);
    retArrays = (copyCSR : retType);
  }
  return retArrays;
}
private proc deepCastToBase(in handle : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool, param isWeightT64 : bool) : CSR_base {
  return (if (handle.desc.isWeighted) then deepCastToBase(handle, isVertexT64, isEdgeT64, isWeightT64, true) else deepCastToBase(handle, isVertexT64, isEdgeT64, isWeightT64, false));
}
private proc deepCastToBase(in handle : CSR_handle, param isVertexT64 : bool, param isEdgeT64 : bool) : CSR_base {
  return (if (handle.desc.isWeightT64) then deepCastToBase(handle, isVertexT64, isEdgeT64, true) else deepCastToBase(handle, isVertexT64, isEdgeT64, false));
}
private proc deepCastToBase(in handle : CSR_handle, param isVertexT64 : bool) : CSR_base {
  return (if (handle.desc.isEdgeT64) then deepCastToBase(handle, isVertexT64, true) else deepCastToBase(handle, isVertexT64, false));
}
//THIS WILL CREATE A COPY
proc deepCastToBase(in handle : CSR_handle) : CSR_base {
  return (if (handle.desc.isVertexT64) then deepCastToBase(handle, true) else deepCastToBase(handle, false));
}

}
