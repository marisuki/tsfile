/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * encoding/simd/bitpack.h 
 * ------------------------------
 * public interface for faster unpack implementation by AVX2.
 * it covers all possible <encode_width, decode_width> pairs. 
 */

#ifndef ENCODING_BITPACK_SIMD_DECODER_H
#define ENCODING_BITPACK_SIMD_DECODER_H

#include <immintrin.h>

#include "encoding/simd/bitpack_common.h"

#ifndef SIMD_BUFFER_SZ
/** 1024B buffer */
#define SIMD_BUFFER_SZ 1024 
#define EPOC_DECODE_SZ(encode_width, decode_width) (SIMD_BUFFER_SZ/decode_width)*encode_width
#endif

namespace storage {

static int ReadEncodeWidth(uint32_t &length, common::ByteStream &buffer) {
    int ret;
    uint32_t len;
    
    ret = common::SerializationUtil::read_var_uint(len, buffer);
    if(RET_FAIL(ret)) 
    {
        return common::E_PARTIAL_READ;
    }

    length = len;

    return common::E_OK;
}


template<psize_t encode_width, psize_t decode_width>
class BitPackDecoderSIMD: public SIMDDecoder<decode_width> {
private:
    Iterator<encode_width, decode_width>         iter_;
    DecodingBuffer                               buffer_;
    BitPackConstants<encode_width, decode_width> constant_;
    
    u8* encode_;
    u32 esz_;
    u8* decode_;

    u8* epos_;
    u8* dpos_;

    bool signal {true};
public:
    BitPackDecoderSIMD() {
        // this->iter_     = Iterator();
        this->buffer_   = DecodingBuffer(SIMD_BUFFER_SZ/encode_width);
        this->decode_   = this->buffer_.buff_;
        this->constant_ = BitPackConstants<encode_width, decode_width>();

        this->esz_      = -1;
    }

    void setInput(u8* input, int bytes) override;
    
    template<typename T>
    void next(T &t) override;

    bool hasNext() override;

    template<typename T> 
    void decodeAll(T* array_ptr) override;

protected:
    inline void decode();
    inline void decode_8();
    inline void decode_16();
    inline void decode_32();
    inline void decode_64();
};

template<psize_t encode_width, psize_t decode_width>
void BitPackDecoderSIMD<encode_width,decode_width>::setInput(u8* input, int bytes) {
    this->encode_   = input;
    this->epos_     = input;
    this->esz_      = bytes;
}

template<psize_t encode_width, psize_t decode_width>
inline void BitPackDecoderSIMD<encode_width, decode_width>::decode() {
    if(encode_width > decode_width) {
        std::cerr << "Illegal cases of encode/decode widths" << std::endl;
        signal = false;
        return;
    }
    if(decode_width == 8) { 
        this->decode_8();
    } 
    else if(decode_width == 16) {
        this->decode_16();
    }
    else if(decode_width == 32) {
        this->decode_32();
    }
    else if(decode_width == 64) {
        this->decode_64();
    }
    else {
        std::cerr << "Illegal cases of decode width" << std::endl;
        signal = false;
        return;
    }
}

template<psize_t encode_width, psize_t decode_width>
inline void BitPackDecoderSIMD<encode_width, decode_width>::decode_8() {
    u32 epoc = EPOC_DECODE_SZ(encode_width, 8); // per-iter of encode in bytes
    int code = 0;
    int add_byte = constant_->get_add_bytes(code);
    int xpos = 0, tot = constant_->total();
    const __m256i mask = _mm256_set1_epi32(((1<<encode_width)-1));
    __m256i reg[4]; 
    int reg_pos = 0;


    constant_->set_target_width(32);
    while(epos_ < encode_ + esz_ && epoc) {
        __m256i loadv = _mm256_loadu2_m128i((__m128i*) (epos_+add_byte), (__m128i*) epos_);
        constant_->restore();
        xpos = 0;
        while(!code && xpos++ < tot) {
            const __m256i sh_const = _mm256_loadu_si256((__m256i*) constant_->get_shuffle_next(code));
            const __m256i mov_const = _mm256_loadu_si256((__m256i*) constant_->get_mov_next(code));
            __m256i shufflev = _mm256_shuffle_epi8(loadv, sh_const);
            shufflev = _mm256_srlv_epi32(shufflev, mov_const);
            reg[reg_pos++] = _mm256_and_si256(shufflev, mask);
            if(reg_pos == 4) {
                reg[0] = _mm256_or_si256(reg[0], _mm256_slli_epi32(reg[1], 8));
                reg[0] = _mm256_or_si256(reg[0], _mm256_slli_epi32(reg[2], 16));
                reg[0] = _mm256_or_si256(reg[0], _mm256_slli_epi32(reg[3], 24));
                _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
                reg_pos = 0;
                dpos_ += 32;
            }
        }
        epos_ += 2*add_byte;
        epoc  -= 2*add_byte;
    }

    if(reg_pos != 0) {
        for(int ref = 1; ref < reg_pos;ref++) {
            reg[0] = _mm256_or_si256(reg[0], _mm256_slli_epi32(reg[ref], 8 * ref));
        }
        _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
    }
}

template<psize_t encode_width, psize_t decode_width>
inline void BitPackDecoderSIMD<encode_width, decode_width>::decode_16() {
    u32 epoc = EPOC_DECODE_SZ(encode_width, 16); // per-iter of encode in bytes
    int code = 0;
    int add_byte = constant_->get_add_bytes(code);
    int xpos = 0;
    int tot = constant_->total();
    const __m256i mask = _mm256_set1_epi32(((1<<encode_width)-1));
    if(code < 0) return;
    __m256i reg[2];
    int reg_pos = 0;

    constant_->set_target_width(32);
    while(epos_ < encode_ + esz_ && epoc) {
        __m256i loadv = _mm256_loadu2_m128i((__m128i*) (epos_ + add_byte), (__m128i*) epos_);
        constant_->restore();
        xpos = 0;
        while(!code && xpos++ < tot) {
            const __m256i sh_const = _mm256_loadu_si256((__m256i*) constant_->get_shuffle_next(code));
            const __m256i mov_const = _mm256_loadu_si256((__m256i*) constant_->get_mov_next(code));
            __m256i shufflev = _mm256_shuffle_epi8(loadv, sh_const);
            shufflev = _mm256_srlv_epi32(shufflev, mov_const);
            reg[reg_pos++] = _mm256_and_si256(shufflev, mask);
            if(reg_pos == 2) {
                reg[0] = _mm256_or_si256(reg[0], _mm256_slli_epi32(reg[1], 16));
                _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
                reg_pos = 0; 
                dpos_ += 32;
            }
        }
        epos_ += 2*add_byte;
        epoc  -= 2*add_byte;
    }

    if(reg_pos != 0) {
        // layout refer
        _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
    }
}

template<psize_t encode_width, psize_t decode_width>
inline void BitPackDecoderSIMD<encode_width, decode_width>::decode_32() {
    u32 epoc = EPOC_DECODE_SZ(encode_width, 32); // per-iter of encode in bytes
    int code = 0;
    int add_byte = constant_->get_add_bytes(code);
    int xpos = 0;
    int tot = constant_->total();
    const __m256i mask = _mm256_set1_epi32(((1<<encode_width)-1));
    if(code < 0) return;
    int reg_pos = 0;

    constant_->set_target_width(32);
    while(epos_ < encode_ + esz_ && epoc) {
        __m256i loadv = _mm256_loadu2_m128i((__m128i*) (epos_+add_byte), (__m128i*) epos_);
        constant_->restore();
        xpos = 0;
        while(!code && xpos++ < tot) {
            const __m256i sh_const = _mm256_loadu_si256((__m256i*) constant_->get_shuffle_next(code));
            const __m256i mov_const = _mm256_loadu_si256((__m256i*) constant_->get_mov_next(code));
            __m256i shufflev = _mm256_shuffle_epi8(loadv, sh_const);
            shufflev = _mm256_srlv_epi32(shufflev, mov_const);
            shufflev = _mm256_and_si256(shufflev, mask);
            _mm256_storeu_si256((__m256i*) dpos_, shufflev);
            dpos_ += 32;
        }
        epos_ += 2*add_byte;
        epoc  -= 2*add_byte;
    }

    if(reg_pos != 0) {
        // layout refer
        _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
    }
}

template<psize_t encode_width, psize_t decode_width>
inline void BitPackDecoderSIMD<encode_width, decode_width>::decode_64() {
    u32 epoc = EPOC_DECODE_SZ(encode_width, 64); // per-iter of encode in bytes
    int code = 0;
    int add_byte = constant_->get_add_bytes(code);
    int xpos = 0;
    int tot = constant_->total();
    const __m256i mask = _mm256_set1_epi64x((1ll<<encode_width)-1);

    constant_->set_target_width(64);
    while(epos_ < encode_ + esz_ && epoc) {
        __m256i loadv = _mm256_loadu2_m128i((__m128i*) (epos_+add_byte), (__m128i*) epos_);
        constant_->restore();
        xpos = 0;
        while(!code && xpos++ < tot) {
            const __m256i sh_const = _mm256_loadu_si256((__m256i*) constant_->get_shuffle_next(code));
            const __m256i mov_const = _mm256_loadu_si256((__m256i*) constant_->get_mov_next(code));
            __m256i shufflev = _mm256_shuffle_epi8(loadv, sh_const);
            shufflev = _mm256_srlv_epi64(shufflev, mov_const);
            shufflev = _mm256_and_si256(shufflev, mask);
            _mm256_storeu_si256((__m256i*) dpos_, shufflev);
            dpos_ += 32;
        }
        epos_ += 2*add_byte;
        epoc  -= 2*add_byte;
    }

    if(reg_pos != 0) {
        // layout refer
        _mm256_storeu_si256((__m256i*) dpos_, reg[0]);
    }
}

template<>
inline void BitPackDecoderSIMD<8, 8>::decode() {
    this->iter_ = PrimaryIterator<8>(this->encode_, this->esz_);
    return;
}

template<>
inline void BitPackDecoderSIMD<16, 16>::decode() {
    this->iter_ = PrimaryIterator<16>(this->encode_, this->esz_);
    return;
}

template<>
inline void BitPackDecoderSIMD<32, 32>::decode() {
    this->iter_ = PrimaryIterator<32>(this->encode_, this->esz_);
    return;
}

template<>
inline void BitPackDecoderSIMD<64, 64>::decode() {
    this->iter_ = PrimaryIterator<64>(this->encode_, this->esz_);
    return;
}

template<>
inline void BitPackDecoderSIMD<8, 16>::decode() {
    u32 epoc = EPOC_DECODE_SZ(8, 16); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<16>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si128((__m128i*) epos_);
            __m256i res = _mm256_cvtepu8_epi16(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 16;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si128((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si128((__m128i*) epos_ + 16);
            __m128i loadv3 = _mm_loadu_si128((__m128i*) epos_ + 32);
            __m128i loadv4 = _mm_loadu_si128((__m128i*) epos_ + 48);
            __m256i res1 = _mm256_cvtepu8_epi16(loadv1);
            __m256i res2 = _mm256_cvtepu8_epi16(loadv2);
            __m256i res3 = _mm256_cvtepu8_epi16(loadv3);
            __m256i res4 = _mm256_cvtepu8_epi16(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 64;
            dpos_ += 128;
        }
    }
}

template<>
inline void BitPackDecoderSIMD<8, 32>::decode() {
    u32 epoc = EPOC_DECODE_SZ(8, 32); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<32>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si64((__m128i*) epos_);
            __m256i res = _mm256_cvtepu8_epi32(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 8;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si64((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si64((__m128i*) epos_ + 8);
            __m128i loadv3 = _mm_loadu_si64((__m128i*) epos_ + 16);
            __m128i loadv4 = _mm_loadu_si64((__m128i*) epos_ + 24);
            __m256i res1 = _mm256_cvtepu8_epi32(loadv1);
            __m256i res2 = _mm256_cvtepu8_epi32(loadv2);
            __m256i res3 = _mm256_cvtepu8_epi32(loadv3);
            __m256i res4 = _mm256_cvtepu8_epi32(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 32;
            dpos_ += 128;
        }
    }
}

template<>
inline void BitPackDecoderSIMD<8, 64>::decode() {
    u32 epoc = EPOC_DECODE_SZ(8, 64); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<64>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si32((__m128i*) epos_);
            __m256i res = _mm256_cvtepu8_epi64(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 4;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si32((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si32((__m128i*) epos_ + 4);
            __m128i loadv3 = _mm_loadu_si32((__m128i*) epos_ + 8);
            __m128i loadv4 = _mm_loadu_si32((__m128i*) epos_ + 12);
            __m256i res1 = _mm256_cvtepu8_epi64(loadv1);
            __m256i res2 = _mm256_cvtepu8_epi64(loadv2);
            __m256i res3 = _mm256_cvtepu8_epi64(loadv3);
            __m256i res4 = _mm256_cvtepu8_epi64(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 16;
            dpos_ += 128;
        }
    }
}

template<>
inline void BitPackDecoderSIMD<16, 32>::decode() {
    u32 epoc = EPOC_DECODE_SZ(16, 32); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<32>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si64((__m128i*) epos_);
            __m256i res = _mm256_cvtepu8_epi32(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 8;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si64((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si64((__m128i*) epos_ + 8);
            __m128i loadv3 = _mm_loadu_si64((__m128i*) epos_ + 16);
            __m128i loadv4 = _mm_loadu_si64((__m128i*) epos_ + 24);
            __m256i res1 = _mm256_cvtepu8_epi32(loadv1);
            __m256i res2 = _mm256_cvtepu8_epi32(loadv2);
            __m256i res3 = _mm256_cvtepu8_epi32(loadv3);
            __m256i res4 = _mm256_cvtepu8_epi32(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 32;
            dpos_ += 128;
        }
    }
}

template<>
inline void BitPackDecoderSIMD<16, 64>::decode() {
    u32 epoc = EPOC_DECODE_SZ(16, 64); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<64>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si64((__m128i*) epos_);
            __m256i res = _mm256_cvtepu16_epi64(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 8;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si64((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si64((__m128i*) epos_ + 8);
            __m128i loadv3 = _mm_loadu_si64((__m128i*) epos_ + 16);
            __m128i loadv4 = _mm_loadu_si64((__m128i*) epos_ + 24);
            __m256i res1 = _mm256_cvtepu16_epi64(loadv1);
            __m256i res2 = _mm256_cvtepu16_epi64(loadv2);
            __m256i res3 = _mm256_cvtepu16_epi64(loadv3);
            __m256i res4 = _mm256_cvtepu16_epi64(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 32;
            dpos_ += 128;
        }
    }
}

template<>
inline void BitPackDecoderSIMD<32, 64>::decode() {
    u32 epoc = EPOC_DECODE_SZ(32, 64); // per-iter of encode in bytes
    this->iter_ = PrimaryIterator<64>(this->encode_, this->esz_);
    if(epos_ + epoc > encode_ + esz_) {
        while(epos_ < encode_ + esz_) {
            __m128i loadv = _mm_loadu_si64((__m128i*) epos_);
            __m256i res = _mm256_cvtepu32_epi64(loadv);
            _mm256_storeu_si256((__m256i*) dpos_, res);
            epos_ += 8;
            dpos_ += 32;
        }
    }
    else {
        for(int i=0;i<epoc;i+=4) {
            __m128i loadv1 = _mm_loadu_si64((__m128i*) epos_);
            __m128i loadv2 = _mm_loadu_si64((__m128i*) epos_ + 8);
            __m128i loadv3 = _mm_loadu_si64((__m128i*) epos_ + 16);
            __m128i loadv4 = _mm_loadu_si64((__m128i*) epos_ + 24);
            __m256i res1 = _mm256_cvtepu32_epi64(loadv1);
            __m256i res2 = _mm256_cvtepu32_epi64(loadv2);
            __m256i res3 = _mm256_cvtepu32_epi64(loadv3);
            __m256i res4 = _mm256_cvtepu32_epi64(loadv4);
            _mm256_storeu_si256((__m256i*) dpos_, res1);
            _mm256_storeu_si256((__m256i*) dpos_ + 32, res2);
            _mm256_storeu_si256((__m256i*) dpos_ + 64, res3);
            _mm256_storeu_si256((__m256i*) dpos_ + 96, res4);
            epos_ += 32;
            dpos_ += 128;
        }
    }
}

}


#endif

