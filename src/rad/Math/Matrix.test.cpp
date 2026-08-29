#include <rad/Math/Matrix.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>

static_assert(std::same_as<rad::Matrix2f, rad::Matrix2<float>>);
static_assert(std::same_as<rad::Matrix2x3d, rad::Matrix2x3<double>>);
static_assert(std::same_as<rad::Matrix3x2i, rad::Matrix3x2<std::int32_t>>);
static_assert(sizeof(rad::Matrix2f) == 4 * sizeof(float));
static_assert(sizeof(rad::Matrix2x3f) == 6 * sizeof(float));
static_assert(sizeof(rad::Matrix3x4f) == 12 * sizeof(float));

TEST(Math, StaticMatrix)
{
    {
        rad::Matrix2d lhs;
        lhs << 1.0, 2.0,
               5.0, 6.0;
        rad::Matrix2d rhs;
        rhs << 3.0, 4.0,
               7.0, 8.0;
        EXPECT_TRUE(rad::HadamardProduct(lhs, rhs).isApprox(lhs.cwiseProduct(rhs)));
    }

    {
        const rad::Vector3d column{1.0, 2.0, 3.0};
        const rad::Vector2d row{4.0, 5.0};
        EXPECT_TRUE(rad::OuterProduct(column, row).isApprox(column * row.transpose()));
    }

    {
        rad::Matrix2x3d matrix;
        matrix << 1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0;
        EXPECT_TRUE(rad::Transpose(matrix).isApprox(matrix.transpose()));
    }

    {
        rad::Matrix2d matrix;
        matrix << 4.0, 7.0,
                  2.0, 6.0;
        EXPECT_DOUBLE_EQ(rad::Determinant(matrix), matrix.determinant());
        EXPECT_DOUBLE_EQ(rad::Determinant(rad::Matrix3d::Identity()),
                         rad::Matrix3d::Identity().determinant());
    }

    {
        rad::Matrix2d matrix;
        matrix << 4.0, 7.0,
                  2.0, 6.0;
        EXPECT_TRUE(rad::Inverse(matrix).isApprox(matrix.inverse()));
    }
}
