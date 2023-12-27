#ifndef _HASH_INCLUDE
#define _HASH_INCLUDE

//MurmurHash3 specialized on input length
//MurmurHash3 was written by Austin Appleby, and is placed in the public domain.
//See https://github.com/aappleby/smhasher

#define _HASH_C1 0xCC9E2D51
#define _HASH_C2 0x1B873593
#define _HASH_C3 0xE6546B64

//helper
#define _HASH_ROTL(X, R) (X << R) | (X >> (32 - R))

uint _fmix32(uint h) {
    h ^= h >> 16;
    h *= 0x85EBCA6B;
    h ^= h >> 13;
    h *= 0xC2B2AE35;
    h ^= h >> 16;

    return h;
}

//Specialized to (1 -> 1) hash
uint hash(uint a, uint seed) {
    uint h1 = seed;

    //only one block; mangle in place
    a *= _HASH_C1;
    a = _HASH_ROTL(a, 15);
    a *= _HASH_C2;

    h1 ^= a;
    h1 = _HASH_ROTL(h1, 13);
    h1 = h1*5 + _HASH_C3;

    //no tail

    //finalize
    h1 ^= 4; //h1 ^= len
    return _fmix32(h1);
}

//Specialized to (2 -> 1) Hash
uint hash(uint a, uint b, uint seed) {
    uint h1 = seed;

    //first block (k1 for i=-2); mangle in place
    a *= _HASH_C1;
    a = _HASH_ROTL(a, 15);
    a *= _HASH_C2;

    h1 ^= a;
    h1 = _HASH_ROTL(h1, 13);
    h1 = h1*5 + _HASH_C3;

    //second block (k1 for i=-1); mangle in place
    b *= _HASH_C1;
    b = _HASH_ROTL(b, 15);
    b *= _HASH_C2;

    h1 ^= b;
    h1 = _HASH_ROTL(h1, 13);
    h1 = h1*5 + _HASH_C3;

    //no tail

    //finalize
    h1 ^= 8; //h1 ^= len
    return _fmix32(h1);
}

//remove consts
#undef _HASH_C1
#undef _HASH_C2
#undef _HASH_C3
#undef _HASH_ROTL

#endif
