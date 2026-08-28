#include <rad/Math/Vector.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

struct NonTrivialScalar
{
    ~NonTrivialScalar() {}
};

template <typename T>
concept ValidVector3Scalar = requires { typename rad::Vector3<T>; };

static_assert(std::same_as<rad::VectorN<float, 3>, rad::Vector3<float>>);
static_assert(std::same_as<rad::VectorX<float>, Eigen::VectorXf>);
static_assert(sizeof(rad::Vector2<float>) == 2 * sizeof(float));
static_assert(sizeof(rad::Vector3<float>) == 3 * sizeof(float));
static_assert(sizeof(rad::Vector4<float>) == 4 * sizeof(float));
static_assert(ValidVector3Scalar<float>);
static_assert(!ValidVector3Scalar<NonTrivialScalar>);

TEST(Math, VectorAngleConversions)
{
    const rad::Vector3<double> degrees{0.0, 90.0, 180.0};
    const rad::Vector3<double> radians = rad::Radians(degrees);

    EXPECT_TRUE(radians.isApprox(rad::Vector3<double>{0.0, rad::HalfPi<double>, rad::Pi<double>}));
    EXPECT_TRUE(rad::Degrees(radians).isApprox(degrees));
}

TEST(Math, DynamicVector)
{
    const rad::VectorX<float> degrees = rad::VectorX<float>::LinSpaced(3, 0.0f, 180.0f);
    const rad::VectorX<float> radians = rad::Radians(degrees);

    ASSERT_EQ(radians.size(), 3);
    EXPECT_TRUE(radians.isApprox(rad::Vector3<float>{0.0f, rad::HalfPi<float>, rad::Pi<float>}));
}

TEST(Math, VectorTrigonometry)
{
    const rad::Vector3<double> values{-0.5, 0.0, 0.5};
    const rad::Vector3<double> positiveValues{1.0, 1.5, 2.0};
    const rad::Vector3<double> x{1.0, -1.0, 1.0};
    const rad::Vector3<double> y{1.0, 1.0, -1.0};

    EXPECT_TRUE(rad::Sin(values).isApprox(values.array().sin().matrix()));
    EXPECT_TRUE(rad::Atan2(y, x).isApprox(y.array().atan2(x.array()).matrix()));
    EXPECT_TRUE(rad::Tanh(values).isApprox(values.array().tanh().matrix()));
    EXPECT_TRUE(rad::Acosh(positiveValues).isApprox(positiveValues.array().acosh().matrix()));
}

TEST(Math, VectorExponentialFunctions)
{
    const rad::Vector3<double> values{1.0, 2.0, 4.0};
    const rad::Vector3<double> exponents{2.0, 3.0, 0.5};

    EXPECT_TRUE(rad::Pow(values, exponents).isApprox(values.array().pow(exponents.array()).matrix()));
    EXPECT_TRUE(rad::Log2(values).isApprox(values.array().log2().matrix()));
    EXPECT_TRUE(rad::Rsqrt(values).isApprox(values.array().rsqrt().matrix()));
}

TEST(Math, VectorCommonFunctions)
{
    const rad::Vector3<double> values{-1.5, -0.5, 2.5};
    const rad::Vector3<double> other{-2.0, 1.0, 2.0};

    EXPECT_TRUE(rad::RoundEven(values).isApprox(rad::Vector3<double>{-2.0, 0.0, 2.0}));
    EXPECT_TRUE(rad::Fract(values).isApprox(rad::Vector3<double>{0.5, 0.5, 0.5}));
    EXPECT_TRUE(rad::Mod(values, 2.0).isApprox(rad::Vector3<double>{0.5, 1.5, 0.5}));
    EXPECT_TRUE(rad::Clamp(values, -1.0, 1.0).isApprox(rad::Vector3<double>{-1.0, -0.5, 1.0}));
    EXPECT_TRUE(rad::Mix(values, other, 0.5).isApprox(rad::Vector3<double>{-1.75, 0.25, 2.25}));
    EXPECT_TRUE(rad::Step(0.0, values).isApprox(rad::Vector3<double>{0.0, 0.0, 1.0}));
    EXPECT_TRUE(rad::SmoothStep(0.0, 1.0, rad::Vector3<double>{-1.0, 0.5, 2.0})
                    .isApprox(rad::Vector3<double>{0.0, 0.5, 1.0}));

    rad::Vector3<double> integralParts;
    EXPECT_TRUE(rad::Modf(values, integralParts).isApprox(rad::Vector3<double>{-0.5, -0.5, 0.5}));
    EXPECT_TRUE(integralParts.isApprox(rad::Vector3<double>{-1.0, 0.0, 2.0}));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    const auto nanMask = rad::IsNaN(rad::Vector3<double>{nan, 0.0, infinity});
    const auto infinityMask = rad::IsInfinite(rad::Vector3<double>{nan, 0.0, infinity});
    EXPECT_TRUE(nanMask.coeff(0));
    EXPECT_TRUE(infinityMask.coeff(2));

    EXPECT_TRUE(rad::Fma(values, other, rad::Vector3<double>::Ones())
                    .isApprox(rad::Vector3<double>{4.0, 0.5, 6.0}));
}

TEST(Math, VectorFrexpLdexp)
{
    const rad::Vector3<double> values{1.0, 2.0, 3.0};
    rad::Vector3<std::int32_t> exponents;
    const rad::Vector3<double> significands = rad::Frexp(values, exponents);

    EXPECT_TRUE(significands.isApprox(rad::Vector3<double>{0.5, 0.5, 0.75}));
    EXPECT_EQ(exponents, rad::Vector3<std::int32_t>(1, 2, 2));
    EXPECT_TRUE(rad::Ldexp(significands, exponents).isApprox(values));
}

TEST(Math, VectorGeometryFunctions)
{
    const rad::Vector3<double> value{3.0, 4.0, 0.0};
    const rad::Vector3<double> x{1.0, 0.0, 0.0};
    const rad::Vector3<double> y{0.0, 1.0, 0.0};
    const rad::Vector3<double> z{0.0, 0.0, 1.0};

    EXPECT_DOUBLE_EQ(rad::Length(value), 5.0);
    EXPECT_TRUE(rad::Cross(x, y).isApprox(z));
    EXPECT_TRUE(rad::Normalize(value).isApprox(rad::Vector3<double>{0.6, 0.8, 0.0}));

    EXPECT_TRUE(rad::FaceForward(z, -z, z).isApprox(z));
    EXPECT_TRUE(rad::Reflect(rad::Vector3<double>{1.0, -1.0, 0.0}, y)
                    .isApprox(rad::Vector3<double>{1.0, 1.0, 0.0}));
    EXPECT_TRUE(rad::Refract(-y, y, 0.5).isApprox(-y));

    const rad::Vector3<double> totalInternalReflection{std::sqrt(3.0) / 2.0, -0.5, 0.0};
    EXPECT_TRUE(rad::Refract(totalInternalReflection, y, 2.0).isZero());
}

TEST(Math, VectorFloatingPointPacking)
{
    EXPECT_EQ(rad::PackUnorm2x16(rad::Vector2<float>{0.0f, 1.0f}), 0xFFFF0000u);
    EXPECT_EQ(rad::PackSnorm2x16(rad::Vector2<float>{-1.0f, 1.0f}), 0x7FFF8001u);
    EXPECT_EQ(rad::PackUnorm4x8(rad::Vector4<float>{0.0f, 1.0f, 0.5f, 0.25f}), 0x4080FF00u);
    EXPECT_EQ(rad::PackSnorm4x8(rad::Vector4<float>{-1.0f, 1.0f, 0.5f, -0.5f}), 0xC0407F81u);

    EXPECT_TRUE(rad::UnpackUnorm2x16(0xFFFF0000u).isApprox(rad::Vector2<float>{0.0f, 1.0f}));
    EXPECT_TRUE(rad::UnpackSnorm2x16(0x7FFF8001u).isApprox(rad::Vector2<float>{-1.0f, 1.0f}));
    EXPECT_TRUE(rad::UnpackUnorm4x8(0xFFAA5500u)
                    .isApprox(rad::Vector4<float>{0.0f, 1.0f / 3.0f, 2.0f / 3.0f, 1.0f}));
    EXPECT_TRUE(rad::UnpackSnorm4x8(0x7F008081u)
                    .isApprox(rad::Vector4<float>{-1.0f, -1.0f, 0.0f, 1.0f}));

    const std::uint32_t packedHalf = rad::PackHalf2x16(rad::Vector2<float>{1.0f, -2.0f});
    EXPECT_EQ(packedHalf, 0xC0003C00u);
    EXPECT_TRUE(rad::UnpackHalf2x16(packedHalf).isApprox(rad::Vector2<float>{1.0f, -2.0f}));

    const rad::Vector2<std::uint32_t> doubleBits{0u, 0x3FF00000u};
    EXPECT_DOUBLE_EQ(rad::PackDouble2x32(doubleBits), 1.0);
    EXPECT_EQ(rad::UnpackDouble2x32(1.0), doubleBits);
}
