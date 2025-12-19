#ifndef DATA_TYPE_H
#define DATA_TYPE_H

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
#define DataTypeBFloat16 12
#define DataTypeFloat8E4M3 13
#define DataTypeFloat8E5M2 14

#if defined(DATA_TYPE_ID)

#if ((DATA_TYPE_ID == DataTypeFloat16) || (DATA_TYPE_ID == DataTypeFloat32) || (DATA_TYPE_ID == DataTypeFloat64) || (DATA_TYPE_ID == DataTypeBFloat16) || (DATA_TYPE_ID == DataTypeFloat8E4M3) || (DATA_TYPE_ID == DataTypeFloat8E5M2))
#define DATA_TYPE_IS_FLOATING_POINT 1
#else
#define DATA_TYPE_IS_FLOATING_POINT 0
#endif

#if ((DATA_TYPE_ID == DataTypeSint8) || (DATA_TYPE_ID == DataTypeSint16) || (DATA_TYPE_ID == DataTypeSint32) || (DATA_TYPE_ID == DataTypeSint64) || (DATA_TYPE_ID == DataTypeUint8) || (DATA_TYPE_ID == DataTypeUint16) || (DATA_TYPE_ID == DataTypeUint32) || (DATA_TYPE_ID == DataTypeUint64))
#define DATA_TYPE_IS_INTEGRAL 1
#else
#define DATA_TYPE_IS_INTEGRAL 0
#endif

#endif // defined(DATA_TYPE_ID)

#endif
