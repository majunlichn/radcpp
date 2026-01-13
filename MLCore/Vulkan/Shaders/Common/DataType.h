#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "Complex.h"

#define DataTypeFloat16 1
#define DataTypeFloat32 2
#define DataTypeFloat64 3
#define DataTypeSint8 4
#define DataTypeSint16 5
#define DataTypeSint32 6
#define DataTypeSint64 7
#define DataTypeUint8 8
#define DataTypeUint16 9
#define DataTypeUint32 10
#define DataTypeUint64 11
#define DataTypeBool 12
#define DataTypeComplex32 13
#define DataTypeComplex64 14
#define DataTypeComplex128 15
#define DataTypeBFloat16 16
#define DataTypeFloat8E4M3 17
#define DataTypeFloat8E5M2 18


#if defined(DATA_TYPE_ID)

#if DATA_TYPE_ID == DataTypeFloat16
#define DATA_TYPE float16_t
#endif
#if DATA_TYPE_ID == DataTypeFloat32
#define DATA_TYPE float32_t
#endif
#if DATA_TYPE_ID == DataTypeFloat64
#define DATA_TYPE float64_t
#endif
#if DATA_TYPE_ID == DataTypeSint8
#define DATA_TYPE int8_t
#endif
#if DATA_TYPE_ID == DataTypeSint16
#define DATA_TYPE int16_t
#endif
#if DATA_TYPE_ID == DataTypeSint32
#define DATA_TYPE int32_t
#endif
#if DATA_TYPE_ID == DataTypeSint64
#define DATA_TYPE int64_t
#endif
#if DATA_TYPE_ID == DataTypeUint8
#define DATA_TYPE uint8_t
#endif
#if DATA_TYPE_ID == DataTypeUint16
#define DATA_TYPE uint16_t
#endif
#if DATA_TYPE_ID == DataTypeUint32
#define DATA_TYPE uint32_t
#endif
#if DATA_TYPE_ID == DataTypeUint64
#define DATA_TYPE uint64_t
#endif
#if DATA_TYPE_ID == DataTypeBool
#define Bool uint8_t
#define DATA_TYPE Bool
#endif
#if DATA_TYPE_ID == DataTypeComplex32
#define DATA_TYPE Complex32
#endif
#if DATA_TYPE_ID == DataTypeComplex64
#define DATA_TYPE Complex64
#endif
#if DATA_TYPE_ID == DataTypeComplex128
#define DATA_TYPE Complex128
#endif
#if DATA_TYPE_ID == DataTypeBFloat16
#extension GL_EXT_bfloat16 : require
#define DATA_TYPE bfloat16_t
#endif
#if DATA_TYPE_ID == DataTypeFloat8E4M3
#extension GL_EXT_float_e4m3 : require
#define DATA_TYPE floate4m3_t
#endif
#if DATA_TYPE_ID == DataTypeFloat8E5M2
#extension GL_EXT_float_e5m2 : require
#define DATA_TYPE floate5m2_t
#endif

#if ((DATA_TYPE_ID == DataTypeComplex32) || (DATA_TYPE_ID == DataTypeComplex64) || (DATA_TYPE_ID == DataTypeComplex128))
#define DATA_TYPE_IS_COMPLEX 1
#else
#define DATA_TYPE_IS_COMPLEX 0
#endif

#if ((DATA_TYPE_ID == DataTypeFloat16) || (DATA_TYPE_ID == DataTypeFloat32) || (DATA_TYPE_ID == DataTypeFloat64) || \
    (DATA_TYPE_ID == DataTypeBFloat16) || (DATA_TYPE_ID == DataTypeFloat8E4M3) || (DATA_TYPE_ID == DataTypeFloat8E5M2))
#define DATA_TYPE_IS_FLOATING_POINT 1
#else
#define DATA_TYPE_IS_FLOATING_POINT 0
#endif

#if ((DATA_TYPE_ID == DataTypeSint8) || (DATA_TYPE_ID == DataTypeSint16) || \
    (DATA_TYPE_ID == DataTypeSint32) || (DATA_TYPE_ID == DataTypeSint64) || \
    (DATA_TYPE_ID == DataTypeUint8) || (DATA_TYPE_ID == DataTypeUint16) || \
    (DATA_TYPE_ID == DataTypeUint32) || (DATA_TYPE_ID == DataTypeUint64))
#define DATA_TYPE_IS_INTEGRAL 1
#else
#define DATA_TYPE_IS_INTEGRAL 0
#endif

#if ((DATA_TYPE_ID == DataTypeSint8) || (DATA_TYPE_ID == DataTypeSint16) || \
    (DATA_TYPE_ID == DataTypeSint32) || (DATA_TYPE_ID == DataTypeSint64))
#define DATA_TYPE_IS_SIGNED_INTEGRAL 1
#else
#define DATA_TYPE_IS_SIGNED_INTEGRAL 0
#endif

#if ((DATA_TYPE_ID == DataTypeUint8) || (DATA_TYPE_ID == DataTypeUint16) || \
    (DATA_TYPE_ID == DataTypeUint32) || (DATA_TYPE_ID == DataTypeUint64))
#define DATA_TYPE_IS_UNSIGNED_INTEGRAL 1
#else
#define DATA_TYPE_IS_UNSIGNED_INTEGRAL 0
#endif

#if (DATA_TYPE_IS_FLOATING_POINT || DATA_TYPE_IS_SIGNED_INTEGRAL)
#define DATA_TYPE_IS_SIGNED 1
#else
#define DATA_TYPE_IS_SIGNED 0
#endif

#if (DATA_TYPE_ID == DataTypeBool)
#define DATA_TYPE_IS_BOOL 1
#else
#define DATA_TYPE_IS_BOOL 0
#endif

#if (DATA_TYPE_ID == DataTypeComplex128)
#define DATA_TYPE_IS_128_BIT 1
#define ELEMENT_SIZE 16
#endif

#if ((DATA_TYPE_ID == DataTypeFloat64) || (DATA_TYPE_ID == DataTypeSint64) || (DATA_TYPE_ID == DataTypeUint64) || (DATA_TYPE_ID == DataTypeComplex64))
#define DATA_TYPE_IS_64_BIT 1
#define ELEMENT_SIZE 8
#endif

#if ((DATA_TYPE_ID == DataTypeFloat32) || (DATA_TYPE_ID == DataTypeSint32) || (DATA_TYPE_ID == DataTypeUint32) || (DATA_TYPE_ID == DataTypeComplex32))
#define DATA_TYPE_IS_32_BIT 1
#define ELEMENT_SIZE 4
#endif

#if ((DATA_TYPE_ID == DataTypeFloat16) || (DATA_TYPE_ID == DataTypeSint16) || (DATA_TYPE_ID == DataTypeUint16) || (DATA_TYPE_ID == DataTypeBFloat16))
#define DATA_TYPE_IS_16_BIT 1
#define ELEMENT_SIZE 2
#endif

#if ((DATA_TYPE_ID == DataTypeSint8) || (DATA_TYPE_ID == DataTypeUint8) || \
    (DATA_TYPE_ID == DataTypeFloat8E4M3) || (DATA_TYPE_ID == DataTypeFloat8E5M2))
#define DATA_TYPE_IS_8_BIT 1
#define ELEMENT_SIZE 1
#endif

#if !defined(COMPUTE_TYPE)
    #if ((DATA_TYPE_ID == DataTypeBFloat16) || (DATA_TYPE_ID == DataTypeFloat8E4M3) || (DATA_TYPE_ID == DataTypeFloat8E5M2))
        #define IS_DATA_TYPE_COMPUTABLE 0
        #define COMPUTE_TYPE float
    #else
        #define IS_DATA_TYPE_COMPUTABLE 1
        #if (DATA_TYPE_ID == DataTypeSint8) || (DATA_TYPE_ID == DataTypeSint16)
            #define COMPUTE_TYPE int32_t
        #elif (DATA_TYPE_ID == DataTypeUint8) || (DATA_TYPE_ID == DataTypeUint16)
            #define COMPUTE_TYPE uint32_t
        #else
            #define COMPUTE_TYPE DATA_TYPE
        #endif
    #endif
#endif

#endif // defined(DATA_TYPE_ID)

#endif // DATA_TYPE_H
