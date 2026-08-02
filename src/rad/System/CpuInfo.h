#pragma once

#include <rad/Core/Platform.h>

// https://github.com/google/cpu_features
#include <cpu_features_macros.h>

// https://github.com/google/cpu_features/blob/main/src/utils/list_cpu_features.c
#if defined(CPU_FEATURES_ARCH_X86)
#include <cpuinfo_x86.h>
#elif defined(CPU_FEATURES_ARCH_ARM)
#include <cpuinfo_arm.h>
#elif defined(CPU_FEATURES_ARCH_AARCH64)
#include <cpuinfo_aarch64.h>
#elif defined(CPU_FEATURES_ARCH_MIPS)
#include <cpuinfo_mips.h>
#elif defined(CPU_FEATURES_ARCH_PPC)
#include <cpuinfo_ppc.h>
#elif defined(CPU_FEATURES_ARCH_S390X)
#include <cpuinfo_s390x.h>
#elif defined(CPU_FEATURES_ARCH_RISCV)
#include <cpuinfo_riscv.h>
#elif defined(CPU_FEATURES_ARCH_LOONGARCH)
#include <cpuinfo_loongarch.h>
#endif

#include <string_view>

namespace rad
{

[[nodiscard]] std::string_view GetCpuBrandString() noexcept;

#if defined(CPU_FEATURES_ARCH_X86)
const cpu_features::X86Info& GetX86Info();
const cpu_features::CacheInfo& GetX86CacheInfo();
#elif defined(CPU_FEATURES_ARCH_ARM)
const cpu_features::ArmInfo& GetArmInfo();
#elif defined(CPU_FEATURES_ARCH_AARCH64)
const cpu_features::Aarch64Info& GetAarch64Info();
#elif defined(CPU_FEATURES_ARCH_MIPS)
const cpu_features::MipsInfo& GetMipsInfo();
#elif defined(CPU_FEATURES_ARCH_PPC)
const cpu_features::PPCInfo& GetPPCInfo();
const cpu_features::PPCPlatformStrings& GetPPCPlatformStrings();
#elif defined(CPU_FEATURES_ARCH_S390X)
const cpu_features::S390XInfo& GetS390XInfo();
const cpu_features::S390XPlatformStrings& GetS390XPlatformStrings();
#elif defined(CPU_FEATURES_ARCH_RISCV)
const cpu_features::RiscvInfo& GetRiscvInfo();
#elif defined(CPU_FEATURES_ARCH_LOONGARCH)
const cpu_features::LoongArchInfo& GetLoongArchInfo();
#endif

} // namespace rad
