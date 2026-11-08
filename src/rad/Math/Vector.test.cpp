#include <rad/Math/Vector.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

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

static_assert(rad::ColVectorType<rad::Vector3f>);
static_assert(rad::ColVectorType<rad::Vector3d>);
static_assert(rad::ColVectorType<rad::VectorX<float>>);
static_assert(rad::ColVectorType<decltype(rad::Vector3f::Zero())>);
static_assert(rad::ColVectorType<decltype(rad::Vector3f::UnitX())>);
static_assert(!rad::ColVectorType<Eigen::RowVector3f>);
static_assert(!rad::ColVectorType<Eigen::RowVectorXf>);
static_assert(!rad::ColVectorType<decltype(rad::Vector3f::Zero().transpose())>);
static_assert(!rad::ColVectorType<Eigen::Matrix3f>);
static_assert(!rad::ColVectorType<Eigen::MatrixXf>);
static_assert(!rad::ColVectorType<float>);
static_assert(rad::FloatingPointVector<rad::VectorX<double>>);
static_assert(rad::FloatingPointVector<rad::Vector3<long double>>);
static_assert(!rad::FloatingPointVector<Eigen::RowVector3d>);
static_assert(rad::VectorOfSize<rad::Vector3f, 3>);
static_assert(!rad::VectorOfSize<rad::Vector3f, 2>);
static_assert(!rad::VectorOfSize<Eigen::RowVector3f, 3>);
static_assert(!rad::ArithmeticVector<Eigen::RowVector3f>);
static_assert(!rad::EqualityComparableVector<Eigen::RowVector3f>);
static_assert(rad::Sint32Vector<rad::Vector3i>);
static_assert(!rad::Sint32Vector<Eigen::Matrix<std::int32_t, 1, 3>>);
static_assert(rad::Uint32Vector<rad::Vector3<std::uint32_t>>);
static_assert(!rad::Uint32Vector<Eigen::Matrix<std::uint32_t, 1, 3>>);
static_assert(rad::BoolVector<rad::Vector3<bool>>);
static_assert(!rad::BoolVector<Eigen::Matrix<bool, 1, 3>>);

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
    EXPECT_TRUE(rad::Mix(values, other, 0.5).isApprox(rad::Vector3<double>{-1.75, 0.25, 2.25}));
    EXPECT_TRUE(rad::Step(0.0, values).isApprox(rad::Vector3<double>{0.0, 0.0, 1.0}));

    // Scalar and per-component clamp bounds permit equal endpoints.
    {
        const rad::Vector3d minimum{-1.0, -0.25, 0.0};
        const rad::Vector3d maximum{0.0, 1.0, 2.0};

        EXPECT_TRUE(rad::Clamp(values, -1.0, 1.0).isApprox(rad::Vector3d{-1.0, -0.5, 1.0}));
        EXPECT_TRUE(rad::Clamp(values, minimum, maximum).isApprox(rad::Vector3d{-1.0, -0.25, 2.0}));
        EXPECT_TRUE(rad::Clamp(values, 0.5, 0.5).isApprox(rad::Vector3d::Constant(0.5)));
        EXPECT_TRUE(rad::Clamp(values, minimum, minimum).isApprox(minimum));
    }

    // Ordered scalar and vector edges interpolate internally and saturate externally.
    {
        EXPECT_TRUE(rad::SmoothStep(0.0, 1.0, rad::Vector3d{-1.0, 0.5, 2.0})
                        .isApprox(rad::Vector3d{0.0, 0.5, 1.0}));
        EXPECT_TRUE(rad::SmoothStep(rad::Vector3d{0.0, 1.0, 2.0}, rad::Vector3d{1.0, 3.0, 6.0},
                                   rad::Vector3d{-1.0, 2.0, 8.0})
                        .isApprox(rad::Vector3d{0.0, 0.5, 1.0}));
    }

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

TEST(Math, VectorPermute)
{
    // Cover the supported index containers and repeated indices.
    {
        const rad::Vector3i values{10, 20, 30};
        const std::array<int, 3> perm{2, 0, 1};
        const int repeatedIndices[]{2, 2, 0};
        const std::vector<unsigned int> vectorIndices{1u, 2u, 0u};
        const std::span<const int> spanIndices(perm);

        static_assert(std::same_as<decltype(rad::Permute(values, perm)), rad::Vector3i>);

        EXPECT_EQ(rad::Permute(values, perm), (rad::Vector3i{30, 10, 20}));
        EXPECT_EQ(rad::Permute(values, repeatedIndices), (rad::Vector3i{30, 30, 10}));
        EXPECT_EQ(rad::Permute(values, vectorIndices), (rad::Vector3i{20, 30, 10}));
        EXPECT_EQ(rad::Permute(values, spanIndices), (rad::Vector3i{30, 10, 20}));
    }

    // Evaluate expressions into an owned result, including self-assignment.
    {
        rad::Vector3i values{10, 20, 30};
        const std::array<int, 3> perm{2, 0, 1};

        EXPECT_EQ(rad::Permute(values + values, perm), (rad::Vector3i{60, 20, 40}));
        values = rad::Permute(values, perm);
        EXPECT_EQ(values, (rad::Vector3i{30, 10, 20}));
    }

    // Boolean components can be reordered without arithmetic.
    {
        const rad::Vector3<bool> mask{true, false, false};
        const std::array<std::size_t, 3> perm{2, 0, 1};

        EXPECT_EQ(rad::Permute(mask, perm), (rad::Vector3<bool>{false, true, false}));
    }

    // Dynamic vectors preserve their size, including empty inputs.
    {
        const rad::VectorX<double> values = rad::Vector4d{1.0, 2.0, 3.0, 4.0};
        const std::vector<Eigen::Index> perm{3, 0, 2, 1};
        const auto result = rad::Permute(values, perm);
        const rad::VectorX<double> empty;
        const std::vector<int> emptyIndices;

        static_assert(std::same_as<decltype(rad::Permute(values, perm)), rad::VectorX<double>>);

        ASSERT_EQ(result.size(), values.size());
        EXPECT_TRUE(result.isApprox(rad::Vector4d{4.0, 1.0, 3.0, 2.0}));
        EXPECT_EQ(rad::Permute(empty, emptyIndices).size(), 0);
    }
}

TEST(Math, VectorComponentExtrema)
{
    // Return scalar extrema and zero-based indices for vectors and expressions.
    {
        const rad::Vector4f values{3.5f, -4.25f, 9.0f, 1.0f};

        static_assert(std::same_as<decltype(rad::MinComponent(values)), float>);
        static_assert(std::same_as<decltype(rad::MaxComponent(values)), float>);
        static_assert(std::same_as<decltype(rad::MinComponentIndex(values)), Eigen::Index>);
        static_assert(std::same_as<decltype(rad::MaxComponentIndex(values)), Eigen::Index>);

        EXPECT_FLOAT_EQ(rad::MinComponent(values), -4.25f);
        EXPECT_FLOAT_EQ(rad::MaxComponent(values), 9.0f);
        EXPECT_EQ(rad::MinComponentIndex(values), 1);
        EXPECT_EQ(rad::MaxComponentIndex(values), 2);
        EXPECT_FLOAT_EQ(rad::MinComponent(-values), -9.0f);
        EXPECT_FLOAT_EQ(rad::MaxComponent(-values), 4.25f);
        EXPECT_EQ(rad::MinComponentIndex(-values), 2);
        EXPECT_EQ(rad::MaxComponentIndex(-values), 1);
    }

    // Integer vectors select the first occurrence of tied extrema.
    {
        const rad::Vector4i values{3, -2, -2, 3};

        static_assert(std::same_as<decltype(rad::MinComponent(values)), std::int32_t>);
        static_assert(std::same_as<decltype(rad::MaxComponent(values)), std::int32_t>);

        EXPECT_EQ(rad::MinComponent(values), -2);
        EXPECT_EQ(rad::MaxComponent(values), 3);
        EXPECT_EQ(rad::MinComponentIndex(values), 1);
        EXPECT_EQ(rad::MaxComponentIndex(values), 0);
    }

    // Dynamic vectors retain first-occurrence behavior across repeated extrema.
    {
        rad::VectorX<float> values(9);
        values << 4.0f, -2.0f, 7.0f, 1.0f, 7.0f, -2.0f, 3.0f, 7.0f, -2.0f;

        EXPECT_FLOAT_EQ(rad::MinComponent(values), -2.0f);
        EXPECT_FLOAT_EQ(rad::MaxComponent(values), 7.0f);
        EXPECT_EQ(rad::MinComponentIndex(values), 1);
        EXPECT_EQ(rad::MaxComponentIndex(values), 2);
    }
}

TEST(Math, VectorProduct)
{
    // Preserve the scalar type and multiply every component, including expressions.
    {
        const rad::Vector3f values{2.0f, -3.0f, 4.0f};

        static_assert(std::same_as<decltype(rad::Product(values)), float>);

        EXPECT_FLOAT_EQ(rad::Product(values), -24.0f);
        EXPECT_FLOAT_EQ(rad::Product(values + values), -192.0f);
        EXPECT_EQ(rad::Product(rad::Vector3i{2, -3, 4}), -24);
        EXPECT_EQ(rad::Product(rad::Vector3i{2, 0, 4}), 0);
    }

    // Dynamic vectors use the multiplicative identity for an empty input.
    {
        const rad::VectorX<double> values = rad::Vector3d{1.5, 2.0, 4.0};
        const rad::VectorX<double> empty;

        EXPECT_DOUBLE_EQ(rad::Product(values), 12.0);
        EXPECT_DOUBLE_EQ(rad::Product(empty), 1.0);
    }
}

TEST(Math, VectorLerp)
{
    const rad::Vector3f from{1.0f, 2.0f, 4.0f};
    const rad::Vector3f to{5.0f, 10.0f, 12.0f};
    const rad::Vector3f t{0.0f, 0.25f, 1.0f};

    static_assert(std::same_as<decltype(rad::Lerp(from, to, 0.25f)), rad::Vector3f>);
    static_assert(std::same_as<decltype(rad::Lerp(from, to, t)), rad::Vector3f>);

    // Scalar factors cover endpoints, interpolation, and unclamped extrapolation.
    {
        EXPECT_TRUE(rad::Lerp(from, to, 0.0f).isApprox(from));
        EXPECT_TRUE(rad::Lerp(from, to, 1.0f).isApprox(to));
        EXPECT_TRUE(rad::Lerp(from, to, 0.25f).isApprox(rad::Vector3f{2.0f, 4.0f, 6.0f}));
        EXPECT_TRUE(rad::Lerp(from, to, 1.5f).isApprox(rad::Vector3f{7.0f, 14.0f, 16.0f}));
    }

    // Vector factors interpolate each component independently.
    {
        EXPECT_TRUE(rad::Lerp(from, to, t).isApprox(rad::Vector3f{1.0f, 4.0f, 12.0f}));
    }

    // Both overloads accept Eigen expressions.
    {
        EXPECT_TRUE(rad::Lerp(from + from, to - to, 0.5f).isApprox(from));
        EXPECT_TRUE(rad::Lerp(from, to, t + t).isApprox(rad::Vector3f{1.0f, 6.0f, 20.0f}));
    }

    // Dynamic double vectors preserve the output type and size.
    {
        const rad::VectorX<double> dynamicFrom = from.cast<double>();
        const rad::VectorX<double> dynamicTo = to.cast<double>();
        const rad::VectorX<double> dynamicT = t.cast<double>();
        const auto scalarResult = rad::Lerp(dynamicFrom, dynamicTo, 0.25);
        const auto vectorResult = rad::Lerp(dynamicFrom, dynamicTo, dynamicT);

        static_assert(std::same_as<decltype(scalarResult), const rad::VectorX<double>>);
        static_assert(std::same_as<decltype(vectorResult), const rad::VectorX<double>>);

        ASSERT_EQ(scalarResult.size(), 3);
        ASSERT_EQ(vectorResult.size(), 3);
        EXPECT_TRUE(scalarResult.isApprox(rad::Vector3d{2.0, 4.0, 6.0}));
        EXPECT_TRUE(vectorResult.isApprox(rad::Vector3d{1.0, 4.0, 12.0}));
    }
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

TEST(Math, VectorLengthSquared)
{
    // Squared length avoids a square root and accepts vector expressions.
    {
        const rad::Vector3d value{3.0, 4.0, 0.0};

        static_assert(std::same_as<decltype(rad::LengthSquared(value)), double>);

        EXPECT_DOUBLE_EQ(rad::LengthSquared(value), 25.0);
        EXPECT_DOUBLE_EQ(rad::LengthSquared(value + value), 100.0);
        EXPECT_DOUBLE_EQ(rad::LengthSquared(rad::Vector3d::Zero()), 0.0);
    }

    // Dynamic vectors retain their scalar type.
    {
        const rad::VectorX<float> value = rad::Vector3f{1.0f, 2.0f, 2.0f};

        static_assert(std::same_as<decltype(rad::LengthSquared(value)), float>);

        EXPECT_FLOAT_EQ(rad::LengthSquared(value), 9.0f);
    }
}

TEST(Math, VectorDistanceSquared)
{
    // Squared distance is symmetric and zero for coincident points.
    {
        const rad::Vector3d p0{1.0, 2.0, 3.0};
        const rad::Vector3d p1{4.0, 6.0, 3.0};

        static_assert(std::same_as<decltype(rad::DistanceSquared(p0, p1)), double>);

        EXPECT_DOUBLE_EQ(rad::DistanceSquared(p0, p1), 25.0);
        EXPECT_DOUBLE_EQ(rad::DistanceSquared(p1, p0), 25.0);
        EXPECT_DOUBLE_EQ(rad::DistanceSquared(p0, p0), 0.0);
        EXPECT_DOUBLE_EQ(rad::DistanceSquared(p0 + p0, p1 + p1), 100.0);
    }

    // Dynamic inputs also accept a zero-vector expression.
    {
        const rad::VectorX<float> value = rad::Vector3f{1.0f, 2.0f, 2.0f};

        static_assert(std::same_as<decltype(rad::DistanceSquared(value, value)), float>);

        EXPECT_FLOAT_EQ(rad::DistanceSquared(value, rad::VectorX<float>::Zero(value.size())), 9.0f);
    }
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
    EXPECT_TRUE(rad::Normalize(rad::Vector3d::Zero()).isZero());

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
