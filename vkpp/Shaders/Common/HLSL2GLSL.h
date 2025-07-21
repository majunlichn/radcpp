#ifndef HLSL2GLSL_H
#define HLSL2GLSL_H

// https://learn.microsoft.com/en-us/windows/uwp/gaming/glsl-to-hlsl-reference

#define float2 vec2
#define float3 vec3
#define float4 vec4

#define bool2 bvec2
#define bool3 bvec3
#define bool4 bvec4

#define int2 ivec2
#define int3 ivec3
#define int4 ivec4

#define float2x2 mat2
#define float3x3 mat3
#define float4x4 mat4

void GroupMemoryBarrier()
{
    groupMemoryBarrier();
}

void GroupMemoryBarrierWithGroupSync()
{
    groupMemoryBarrier();
    barrier();
}

void DeviceMemoryBarrier()
{
    memoryBarrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
}

void DeviceMemoryBarrierWithGroupSync()
{
    memoryBarrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    barrier();
}

void AllMemoryBarrier()
{
    groupMemoryBarrier();
    memoryBarrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    memoryBarrierShared();
}

void AllMemoryBarrierWithGroupSync()
{
    groupMemoryBarrier();
    memoryBarrier();
    memoryBarrierBuffer();
    memoryBarrierImage();
    memoryBarrierShared();
    barrier();
}

float atan2(float y, float x)
{
    return atan(y, x);
}

#define ddx dFdx
#define ddy dFdy
#define ddx_coarse dFdxCoarse
#define ddy_coarse dFdyCoarse
#define ddx_fine dFdxFine
#define ddy_fine dFdyFine

#define EvaluateAttributeAtCentroid interpolateAtCentroid
#define EvaluateAttributeAtSample interpolateAtSample
#define EvaluateAttributeSnapped interpolateAtOffset

#define frac fract
#define lerp mix
#define mad fma

#define saturate(x) clamp(x, 0.0, 1.0)

int asint(float x)
{
    floatBitsToInt(x);
}

uint asuint(float x)
{
    floatBitsToUint(x);
}

#if defined(GL_EXT_shader_explicit_arithmetic_types_int64) && defined(GL_EXT_shader_explicit_arithmetic_types_float64)
void asuint(in double value, out uint lowbits, out uint highbits)
{
    u32vec2 bits = unpack32(doubleBitsToUint64(x));
    lowbits = bits[0];
    highbits = bits[1];
}
#endif

float asfloat(int x)
{
    intBitsToFloat(x);
}

float asfloat(uint x)
{
    uintBitsToFloat(x);
}

#endif // HLSL2GLSL_H
