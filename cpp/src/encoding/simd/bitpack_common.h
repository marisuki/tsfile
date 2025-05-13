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

#ifndef BITPACK_SIMD_DECODER_COMMON_H
#define BITPACK_SIMD_DECODER_COMMON_H

#include "encoding/simd/bitpack_offsets.h"
#include "encoding/simd/simd_common.h"

namespace storage {

// #define BITPACK_BUFFER_READ_SZ  102400

template<psize_t width>
class PrimaryIterator: public Iterator<width> {
private:
    u8* buffer_ {nullptr};
    u8* end_ {nullptr};
    u8* curr_ {nullptr};

public:
    /** buffer of decoded data, size in bytes */
    PrimaryIterator(u8* buffer, u32 size):
        buffer_(buffer),
        end_(buffer+size),
        curr_(buffer) {}
    
    void reset() override {
        curr_ = buffer_;
    }

    template<typename T>
    void next(T &x) override {
        x = ((T*) curr_)[0];
        curr_ += width/8;
    }

    bool hasNext() override {
        return curr_ < end_;
    }
};

template<psize_t width>
class BitPackIterator: public Iterator<width> {
private: 
    u32 pos_;
    u32 size_;
    u8* buffer_;

public:
    BitPackIterator(u8* buffer): buffer_(buffer) {
        reset();
    } 

    void reset() override {
        pos_ = size_ = 0;
    }

    template<typename T>
    void next(T &x) override {
        x = ((T*) buffer_)[pos_];
        pos_ ++;
    }

    bool hasNext() override {
        #ifdef DEBUG 
        #  error  Unimplemented error
        #else
            assert(false);
        #endif 
    }
};


template<psize_t encode_width, psize_t decode_width>
class BitPackConstants {
private:
    int     spos_;
    int     mpos_;
    int     tot_;

private:
    void init();

public:
    BitPackConstants()
    {   
        this->spos_ = this->mpos_ = 0; 
        init();
    }

    void  set_target_width(int targetWidth) { this->targetWidth_ = targetWidth;}
    void  restore() { this->spos_ = this->mpos_ = 0; }
    void* get_shuffle_start(int &code);
    void* get_shuffle_next(int &code);
    int   total();
    bool  shuffle_has_next(int &code);
    void* get_mov_start(int &code);
    void* get_mov_next(int &code);
    bool  mov_has_next(int &code);
    int   get_add_bytes(int &code);
    ~BitPackConstants() {}
};


template<psize_t encode_width, psize_t decode_width>
void BitPackConstants<encode_width, decode_width>::init() {
    switch(decode_width) {
        case 8: {
            this->tot_ = 
                storage::constant::bp8_position[encode_width] - storage::constant::bp8_position[encode_width-1];
            return;
        } 
        case 16: {
            this->tot_ = 
                storage::constant::bp16_position[encode_width] - storage::constant::bp16_position[encode_width-1];
            return;
        }
        case 32: {
            this->tot_ = 
                storage::constant::bp32_position[encode_width] - storage::constant::bp32_position[encode_width-1];
            return;
        }
        case 64: {
            this->tot_ = 
                storage::constant::bp64_position[encode_width] - storage::constant::bp64_position[encode_width-1];
            return;
        }
        default: 
            this->tot_ = -1; 
            return;
    }
}

template<psize_t encode_width, psize_t decode_width>
int BitPackConstants<encode_width, decode_width>::get_add_bytes(int &code) {
    // code = 0;
    switch(decode_width) {
        case 8: return storage::constant::bp8_add_bytes[decode_width-1];
        case 16: return storage::constant::bp16_add_bytes[decode_width-1];
        case 32: return storage::constant::bp32_add_bytes[decode_width-1];
        case 64: return storage::constant::bp64_add_bytes[decode_width-1];
        default: code += -1; return -1;
    }
}

template<psize_t encode_width, psize_t decode_width>
void* BitPackConstants<encode_width, decode_width>::get_shuffle_start(int &code) {
    // code = 0;
    switch(decode_width) {
        case 8: {
            if(encode_width >8) {code += -1; return nullptr;}
            else {
                // spos = 1;
                return (void*)(storage::constant::shuffle8[storage::constant::bp8_position[encode_width-1]]);
            }
        } 
        case 16: {
            if(encode_width >16) {code += -1; return nullptr;}
            else {
                // spos = 1;
                return (void*)(storage::constant::shuffle16[storage::constant::bp16_position[encode_width-1]]);
            }
        }
        case 32: {
            if(encode_width >32) {code += -1; return nullptr;}
            else {
                // spos = 1;
                return (void*)(storage::constant::shuffle32[storage::constant::bp32_position[encode_width-1]]);
            }
        }
        case 64: {
            if(encode_width >64) {code += -1; return nullptr;}
            else {
                // spos = 1;
                return (void*)(storage::constant::shuffle64[storage::constant::bp64_position[encode_width-1]]);
            }
        } default: {code += -1; return nullptr;}
    }
}

template<psize_t encode_width, psize_t decode_width>
int BitPackConstants<encode_width, decode_width>::total() {
    return this->tot_;
}

template<psize_t encode_width, psize_t decode_width>
void* BitPackConstants<encode_width, decode_width>::get_shuffle_next(int &code) {
    switch(decode_width) {
        case 8: {
            return (void*)(storage::constant::shuffle8[storage::constant::bp8_position[encode_width-1]+spos_++]);
        } 
        case 16: {
            return (void*)(storage::constant::shuffle16[storage::constant::bp16_position[encode_width-1]+spos_++]);
        }
        case 32: {
            return (void*)(storage::constant::shuffle32[storage::constant::bp32_position[encode_width-1]+spos_++]);
        }
        case 64: {
            return (void*)(storage::constant::shuffle64[storage::constant::bp64_position[encode_width-1]+spos_++]);
        }
        default: {code += -1; return nullptr;}
    }
}

template<psize_t encode_width, psize_t decode_width>
bool BitPackConstants<encode_width, decode_width>::shuffle_has_next(int &code) {
    switch(decode_width) {
        case 8: {
            if(encode_width >8) {code += -1; return false;}
            else {
                return spos_ < tot_;
            }
        }
        case 16: {
            if(encode_width >16) {code += -1; return false;}
            else {
                return spos_ < tot_;
            }
        }
        case 32: {
            if(encode_width >32) {code += -1; return false;}
            else {
                return spos_ < tot_;
            }
        }
        case 64: {
            if(encode_width >64) {code += -1; return false;}
            else {
                return spos_ < tot_;
            }
        }
        default: {code += -1; return false;}
    }
}

template<psize_t encode_width, psize_t decode_width>
void* BitPackConstants<encode_width, decode_width>::get_mov_start(int &code) {
    // code = 0;
    switch(decode_width) {
        case 8: {
            if(encode_width >8) {code += -1; return nullptr;}
            else {
                // mpos = 1;
                return (void*)((storage::constant::mov8[storage::constant::bp8_position[encode_width-1]]));
            }
        } 
        case 16: {
            if(encode_width >16) {code += -1; return nullptr;}
            else {
                // mpos = 1;
                return (void*)((storage::constant::mov16[storage::constant::bp16_position[encode_width-1]]));
            }
        }
        case 32: {
            if(encode_width >32) {code += -1; return nullptr;}
            else {
                // mpos = 1;
                return (void*)((storage::constant::mov32[storage::constant::bp32_position[encode_width-1]]));
            }
        }
        case 64: {
            if(encode_width >64) {code += -1; return nullptr;}
            else {
                // mpos = 1;
                return (void*)((storage::constant::mov64[storage::constant::bp64_position[encode_width-1]]));
            }
        } default: { code += -1; return nullptr;}
    }
}

template<psize_t encode_width, psize_t decode_width>
void* BitPackConstants<encode_width, decode_width>::get_mov_next(int &code) {
    // code = 0;
    switch(decode_width) {
        case 8: {
            return (void*)(storage::constant::mov8[storage::constant::bp8_position[encode_width-1]+mpos_++]);
        } 
        case 16: {
            return (void*)(storage::constant::mov16[storage::constant::bp16_position[encode_width-1]+mpos_++]);
        }
        case 32: {
            return (void*)(storage::constant::mov32[storage::constant::bp32_position[encode_width-1]+mpos_++]);
        }
        case 64: {
            return (void*)(storage::constant::mov64[storage::constant::bp64_position[encode_width-1]+mpos_++]);
        }
        default: {code += -1; return nullptr;}
    }
}

template<psize_t encode_width, psize_t decode_width>
bool BitPackConstants<encode_width, decode_width>::mov_has_next(int &code) {
    switch(decode_width) {
        case 8: {
            if(encode_width >8) {code += -1; return false;}
            else {
                return mpos_ < tot_;
            }
        }
        case 16: {
            if(encode_width >16) {code += -1; return false;}
            else {
                return mpos_ < tot_;
            }
        }
        case 32: {
            if(encode_width >32) {code += -1; return false;}
            else {
                return mpos_ < tot_;
            }
        }
        case 64: {
            if(encode_width >64) {code += -1; return false;}
            else {
                return mpos_ < tot_;
            }
        }
        default: {code += -1; return false;}
    }
}


}
#endif 