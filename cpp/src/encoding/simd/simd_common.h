#ifndef SIMD_COMMON_H
#define SIMD_COMMON_H

#include <errno.h>
#include <assert.h>
#include <iostream>

#include "common/allocator/byte_stream.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace storage {

typedef int32_t psize_t;

#define u8  uint8_t
#define s8  int8_t
#define u16 uint16_t
#define s16 int16_t
#define u32 uint32_t
#define s32 int32_t
#define u64 uint64_t
#define s64 int64_t

#define __register register
#define __align(x) __attribute__((ALIGNED(x)))

template<psize_t width>
class Iterator {
public:
    virtual void reset() {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    template<typename T>
    void next(T &x) {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    virtual bool hasNext() {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }
};


struct DecodingBuffer {
public:
    u8* buff_;
    int sz_;
    DecodingBuffer(int bytes) {
        buff_ = (u8*) malloc(bytes);
        sz_ = bytes;
    }
    u8 at(int loc_byte) {
        if(loc_byte < sz_) return buff_[loc_byte];
        else return buff_[sz_-1];
    }
};

template<psize_t decode_width>
class SIMDDecoder {
public:
    virtual void setInput(u8* input, int bytes) {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    virtual bool hasNext() {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    template<typename T> 
    void next(T &t) {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    template<typename T> 
    void decodeAll(T* array_ptr) {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }

    
    virtual Iterator<decode_width> iterator() {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }
};

}

#endif
