#pragma once

#include <rad/Core/Platform.h>
#include <rad/Core/Integer.h>
#include <cfenv>
#include <cfloat>
#include <cmath>
#include <limits>

namespace rad
{

float Normalize(float value, float min, float max);
uint8_t QuantizeUnorm8(float normalized);
uint16_t QuantizeUnorm16(float normalized);
uint32_t QuantizeUnorm32(double normalized);
float DequantizeUnorm8(uint8_t quantized);
float DequantizeUnorm16(uint16_t quantized);
float DequantizeUnorm32(uint32_t quantized);

} // namespace rad
