#ifndef PACKING_DATA_H
#define PACKING_DATA_H

#ifdef GLSL
#include "HLSL2GLSL.h"
#endif

// https://www.elopezr.com/the-art-of-packing-data/

// Normalized Data

uint PackFloat4ToRGBA8Unorm(float4 value)
{
#if defined(GLSL)
    return packUnorm4x8(value);
#elif defined(HLSL) && __SHADER_TARGET_MAJOR >= 6 && __SHADER_TARGET_MINOR >= 6
    uint4 ivalue = uint4(value * 255.0 + 0.5);
    return pack_u8(uvalue);
#else
    uint4 uvalue = uint4(value * 255.0 + 0.5);
    return (uvalue.a << 24) | (uvalue.b << 16) | (uvalue.g << 8) | uvalue.r;
#endif
}

float4 UnpackRGBA8UnormToFloat4(uint packed)
{
#if defined(GLSL)
    return unpackUnorm4x8(value);
#elif defined(HLSL) && __SHADER_TARGET_MAJOR >= 6 && __SHADER_TARGET_MINOR >= 6
    return float4(unpack_u8u32(packed)) / 255.0;
#else
    uint ri = packed & 0xff;
    uint gi = (packed >> 8) & 0xff;
    uint bi = (packed >> 16) & 0xff;
    uint ai = packed >> 24;
    return float4(ri, gi, bi, ai) / 255.0;
#endif
}

uint PackFloat2ToRGBA16Unorm(float2 value)
{
    uint2 uvalue = uint2(value * 65535.0 + 0.5);
    return (uvalue.g << 16) | uvalue.r;
}

float2 UnpackRGBA16UnormToFloat2(uint packed)
{
    uint ri = packed & 0xffff;
    uint gi = packed >> 16;
    return float2(ri, gi) / 65535.0;
}

uint PackFloat4ToRGB10A2Unorm(float4 value)
{
    uint3 rgbi = uint3(value.rgb * 1023.0 + 0.5);
    uint ai = uint(value.a * 3.0 + 0.5);
    return (ai << 30) | (rgbi.b << 20) | (rgbi.g << 10) | rgbi.r;
}

float4 UnpackRGB10A2UnormToFloat4(uint packed)
{
    uint ri = packed & 0x3ff;
    uint gi = (packed >> 10) & 0x3ff;
    uint bi = (packed >> 20) & 0x3ff;
    uint ai = packed >> 30;
    return float4(float3(ri, gi, bi) / 1023.0, ai / 3.0);
}

// Signed Normalized Data

// Assume input is in the -1, 1 range, for both packing and unpacking. Clamp accordingly if the input is unknown
uint PackFloat4ToRGBA8Snorm(float4 value)
{
#if defined(GLSL)
    return packSnorm4x8(value);
#elif defined(HLSL) && __SHADER_TARGET_MAJOR >= 6 && __SHADER_TARGET_MINOR >= 6
    int4 ivalue = int4(round(value * 127.0));
    return (uint)pack_s8(ivalue);
#else
    int4 ivalue = int4(round(value * 127.0));
    return (uint)((ivalue.a << 24) | (ivalue.b << 16) | (ivalue.g << 8) | ivalue.r);
#endif
}

float4 UnpackRGBA8SnormToFloat4(uint packed)
{
#if defined(GLSL)
    return unpackSnorm4x8(value);
#else
    int ri = (int)(packed << 24) >> 24;
    int gi = (int)(packed << 16) >> 24;
    int bi = (int)(packed << 8) >> 24;
    int ai = (int)(packed << 0) >> 24;
    return float4(ri, gi, bi, ai) / 127.0;
#endif
}

// Bitfields

// These are emulations of the low-level RDNA instruction,
// if you have access to an intrinsic you can replace it.

uint bfe_b32_emu(uint value, uint offset, uint bits)
{
    uint mask = (1U << bits) - 1U;
    uint shiftValue = value >> offset;
    return shiftValue & mask;
}

uint bfi_b32_emu(uint value, uint preserveMask, uint insertMask)
{
    return (value & preserveMask) | (~value & insertMask);
}

// Extract bits numbers of bits at an offset from value
uint BitfieldExtract(uint value, uint offset, uint bits)
{
#ifdef GLSL
    bitfieldExtract(value, offset, bits);
#else
    return bfe_b32_emu(value, offset, bits);
#endif
}

// Insert low bitCount bits of insert at an offset in base
uint BitfieldInsert(uint value, uint insert, uint offset, uint bits)
{
#ifdef GLSL
    bitfieldInsert(value, insert, offset, bits);
#else
    uint preserveMask = ~(~(0xffffffffU << bits) << offset);
    uint insertMask = insert << offset;
    return bfi_b32_emu(value, preserveMask, insertMask);
#endif
}

uint PackFloat4ToRGB10A2UnormOpt(float4 value)
{
    uint3 rgbi = uint3(value.rgb * 1023.0 + 0.5);
    uint ai = uint(value.a * 3.0 + 0.5);
    uint result = rgbi.x;
    result = BitfieldInsert(result, rgbi.y, 20, 10);
    result = BitfieldInsert(result, rgbi.z, 30, 10);
    result = BitfieldInsert(result, ai, 32, 2);
    return result;
}

uint UnpackRGB10A2UnormToFloat4Opt(uint packed)
{
    uint ri = BitfieldExtract(packed, 0, 10);
    uint gi = BitfieldExtract(packed, 10, 10);
    uint bi = BitfieldExtract(packed, 20, 10);
    uint ai = BitfieldExtract(packed, 30, 2);
    return float4(ri / 1023.0f, gi / 1023.0f, bi / 1023.0f, ai / 3.0f);
}

// Floating-Point Data

uint PackFloat3ToRG11B10F(float3 value)
{
    uint3 packed16 = f32tof16(value);
    uint ri = (packed16.r & 0x7ff0) << 17;
    uint gi = (packed16.g & 0x7ff0) << 6;
    uint bi = (packed16.b & 0x7fe0) >> 5;
    return ri | gi | bi;
}

float3 UnpackRG11B10FToFloat3(uint packed)
{
    uint ri = (packed >> 17) & 0x7ff0;
    uint gi = (packed >> 6) & 0x7ff0;
    uint bi = (packed << 5) & 0x7fe0;
    return f16tof32(uint3(ri, gi, bi));
}

uint PackFloat3ToRGB9E5F(float3 value)
{
    // Clamp the channels to an expressible range:
    const float MaxValue = asfloat(0x477f8000); // 0.ff x 2^+15

    float3 clampedValue = clamp(value, 0.0f, MaxValue);

    // Compute maximum channel of clamped value:
    float maxChannel = max(clampedValue.x, max(clampedValue.y, clampedValue.z));

    // "bias" has to have the biggest exponent plus 15 (and nothing in the mantissa). When added to the three channels,
    // it shifts the explicit '1' and the 8 most significant mantissa bits into the low 9 bits.
    // IEEE rules of float addition will round rather than truncate the discarded bits.
    // Channels with smaller natural exponents will be shifted further to the right (discarding more bits).

    // Reinterpret the maximum channel as a uint:
    uint maxChannelUint = asuint(maxChannel);

    // Bias the exponent by 15, and shift the mantissa into the low 9 bits:
    // 0x07804000 == 0b 0 00001111 00000000100000000000000
    uint maxChannelBiased = maxChannelUint + 0x07804000;

    // Keep only the exponent of the resulting float:
    // 0x7F800000 == 0b 0 11111111 00000000000000000000000
    uint biasu = maxChannelBiased & 0x7F800000;

    // Shift the bias 4 positions to keep only 5 bits:
    uint biass = biasu << 4;
    uint exponent = biass + 0x10000000;

    // Turn bias back into float and add to the original value to shift bits into the right places:
    float bias = asfloat(biasu);

    float3 clampedBiased = clampedValue + bias;
    uint3 rgb = (uint3&)(clampedBiased);

    return exponent | rgb.z << 18 | rgb.y << 9 | (rgb.x & 0x1ff);
}

uint UnpackRGB9E5FToFloat3(uint packed)
{
    float3 rgb = uint3(p, p >> 9, p >> 18) & uint3(0x1ff);
    return ldexp(rgb, (int)(p >> 27) - 24);
}

// Packing Normals

float3 PackNormalTrivial(float3 normal)
{
    return normal * 0.5 + 0.5;
}

float3 UnpackNormalTrivial(float3 packed)
{
    return packed * 2.0 - 1.0;
}

// CryEngine 3 Stereographic
// https://en.wikipedia.org/wiki/Stereographic_projection
float2 PackNormalStereographic(float3 normal)
{
    float2 packed = normal.xy / (1.0f - normal.z);
    packed = packed * 0.5f + 0.5f;
    return packed;
}

float3 UnpackNormalStereographic(float2 packed)
{
    float2 unpacked = packed * 2.0f - 1.0f;
    float dotUnpacked = dot(unpacked, unpacked);
    float denominator = 1.0f + dotUnpacked;
    float3 result;
    result.xy = 2.0f * unpacked.xy / denominator;
    result.z = (-1.0f + dotUnpacked) / denominator;
    return normalize(result);
}

// Octahedral is the most popular way of encoding to encode world space normals in the graphics world as far as I'm aware.

float2 PackNormalOctahedral(float3 normal)
{
    normal /= (abs(normal.x) + abs(normal.y) + abs(normal.z));
    normal.xy =
            normal.z >= 0.0 ? 
            normal.xy :
            (1.0 - abs(normal.yx)) * select(normal.xy >= 0.0, 1.0, -1.0);
    normal.xy = normal.xy * 0.5 + 0.5;
    return normal.xy;
}

float3 UnpackNormalOctahedral(float2 packed)
{
    packed = packed * 2.0 - 1.0;

    // https://twitter.com/Stubbesaurus/status/937994790553227264
    float3 n = float3(packed.xy, 1.0 - abs(packed.x) - abs(packed.y));
    float t = saturate(-n.z);
    n.xy += select(n.xy >= 0.0, -t, t);
    return normalize(n);
}

#endif // PACKING_DATA_H
