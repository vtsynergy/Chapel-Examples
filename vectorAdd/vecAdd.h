#ifndef REAL_TYPE
#define REAL_TYPE float
#endif

extern "C" {
  void vecAdd(REAL_TYPE * A, REAL_TYPE * B, REAL_TYPE * C, int32_t lo, int32_t hi, int32_t nelem);
}
