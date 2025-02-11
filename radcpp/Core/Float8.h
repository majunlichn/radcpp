#pragma once

#include <radcpp/Core/Float.h>

namespace rad
{

float fp8e4m3fn_to_fp32_value(uint8_t input);
uint8_t fp8e4m3fn_from_fp32_value(float f);
float fp8e5m2_to_fp32_value(uint8_t input);
uint8_t fp8e5m2_from_fp32_value(float f);

} // namespace rad
