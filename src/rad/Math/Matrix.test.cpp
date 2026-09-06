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
    using namespace rad;

    {
        Matrix2d lhs;
        lhs << 1.0, 2.0,
               5.0, 6.0;
        Matrix2d rhs;
        rhs << 3.0, 4.0,
               7.0, 8.0;
        EXPECT_TRUE(HadamardProduct(lhs, rhs).isApprox(lhs.cwiseProduct(rhs)));
    }

    {
        const Vector3d column{1.0, 2.0, 3.0};
        const Vector2d row{4.0, 5.0};
        EXPECT_TRUE(OuterProduct(column, row).isApprox(column * row.transpose()));
    }

    {
        Matrix2x3d matrix;
        matrix << 1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0;
        EXPECT_TRUE(Transpose(matrix).isApprox(matrix.transpose()));
    }

    {
        Matrix2d matrix;
        matrix << 4.0, 7.0,
                  2.0, 6.0;
        EXPECT_DOUBLE_EQ(Determinant(matrix), matrix.determinant());
        EXPECT_DOUBLE_EQ(Determinant(Matrix3d::Identity()), Matrix3d::Identity().determinant());
    }

    {
        Matrix2d matrix;
        matrix << 4.0, 7.0,
                  2.0, 6.0;
        EXPECT_TRUE(Inverse(matrix).isApprox(matrix.inverse()));
    }
}

TEST(Math, MatrixTransform)
{
    using namespace rad;

    const Matrix4d identity = Matrix4d::Identity();
    const Vector4d origin{0.0, 0.0, 0.0, 1.0};
    const Vector4d xPoint{1.0, 0.0, 0.0, 1.0};
    const Vector4d yPoint{0.0, 1.0, 0.0, 1.0};
    const Vector4d zPoint{0.0, 0.0, 1.0, 1.0};
    const Vector3d xAxis{1.0, 0.0, 0.0};
    const Vector3d yAxis{0.0, 1.0, 0.0};
    const Vector3d zAxis{0.0, 0.0, 1.0};

    // Translate: points move, directions (w = 0) do not.
    {
        const Vector3d translation{1.0, 2.0, 3.0};
        const Matrix4d translated = Translate(identity, translation);
        const Vector4d point{4.0, 5.0, 6.0, 1.0};
        const Vector4d direction{4.0, 5.0, 6.0, 0.0};

        Matrix4d reference = Matrix4d::Identity();
        reference.col(3).head(3) = translation;
        EXPECT_TRUE(translated.isApprox(reference));
        EXPECT_TRUE((translated * origin).isApprox(Vector4d{1.0, 2.0, 3.0, 1.0}));
        EXPECT_TRUE((translated * point).isApprox(Vector4d{5.0, 7.0, 9.0, 1.0}));
        EXPECT_TRUE((translated * direction).isApprox(direction));
        EXPECT_TRUE(Translate(identity, Vector3d::Zero()).isApprox(identity));
    }

    // Rotate: right-handed quarter-turns about the principal axes.
    {
        const Matrix4d rotateX = Rotate(identity, HalfPi<double>, xAxis);
        const Matrix4d rotateY = Rotate(identity, HalfPi<double>, yAxis);
        const Matrix4d rotateZ = Rotate(identity, HalfPi<double>, zAxis);

        EXPECT_TRUE((rotateX * yPoint).isApprox(zPoint));
        EXPECT_TRUE((rotateX * zPoint).isApprox(Vector4d{0.0, -1.0, 0.0, 1.0}));
        EXPECT_TRUE((rotateY * zPoint).isApprox(xPoint));
        EXPECT_TRUE((rotateY * xPoint).isApprox(Vector4d{0.0, 0.0, -1.0, 1.0}));
        EXPECT_TRUE((rotateZ * xPoint).isApprox(yPoint));
        EXPECT_TRUE((rotateZ * yPoint).isApprox(Vector4d{-1.0, 0.0, 0.0, 1.0}));

        EXPECT_TRUE(Rotate(identity, 0.0, zAxis).isApprox(identity));
        EXPECT_TRUE(Rotate(identity, HalfPi<double>, Vector3d::Zero()).isApprox(identity));
        EXPECT_TRUE(Rotate(rotateZ, -HalfPi<double>, zAxis).isApprox(identity));
    }

    // Rotate: axis is invariant; lengths and orientation are preserved.
    {
        const Vector3d axis{2.0, 2.0, 2.0};
        const Matrix4d rotated = Rotate(identity, Pi<double>, axis);
        const Vector3d unit = axis.normalized();
        const Vector4d onAxis{unit.x(), unit.y(), unit.z(), 1.0};
        EXPECT_TRUE((rotated * onAxis).isApprox(onAxis));

        const Vector4d direction{1.0, 2.0, 3.0, 0.0};
        const Vector4d rotatedDirection = rotated * direction;
        EXPECT_NEAR(rotatedDirection.head(3).norm(), direction.head(3).norm(), 1e-12);

        const Matrix3d rotation3x3 = rotated.block(0, 0, 3, 3);
        EXPECT_NEAR(rotation3x3.determinant(), 1.0, 1e-12);

        Matrix3d reference;
        reference << -1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0,
                     2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0,
                     2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0;
        EXPECT_TRUE(rotation3x3.isApprox(reference));
    }

    // Rotate (2D and 3D in-plane): scalar-angle form is the top-left 2x2 block.
    {
        const Matrix3d identity2d = Matrix3d::Identity();
        const Matrix4d inPlane3d = Rotate(identity, HalfPi<double>);
        const Vector4d inPlanePoint = inPlane3d * xPoint;

        Matrix3d reference2d;
        reference2d << 0.0, -1.0, 0.0,
                       1.0,  0.0, 0.0,
                       0.0,  0.0, 1.0;
        const Matrix3d quarterTurn2d = Rotate(identity2d, HalfPi<double>);
        const Vector3d rotated2dPoint = quarterTurn2d * Vector3d{1.0, 0.0, 1.0};

        EXPECT_TRUE(quarterTurn2d.isApprox(reference2d));
        EXPECT_TRUE(rotated2dPoint.isApprox(Vector3d{0.0, 1.0, 1.0}));
        EXPECT_TRUE(Rotate(quarterTurn2d, -HalfPi<double>).isApprox(identity2d));

        Matrix4d reference4d = Matrix4d::Identity();
        reference4d.block(0, 0, 2, 2) = reference2d.block(0, 0, 2, 2);
        EXPECT_TRUE(inPlane3d.isApprox(reference4d));
        EXPECT_TRUE(inPlanePoint.isApprox(yPoint));
    }

    // Translate and Scale in 2D: affine transforms of 3x3 matrices.
    {
        const Matrix3d identity2d = Matrix3d::Identity();
        const Vector2d translation2d{1.0, 2.0};
        const Vector2d scaling2d{2.0, 3.0};
        const Vector3d point2d{4.0, 5.0, 1.0};

        const Vector3d translated2dPoint = Translate(identity2d, translation2d) * point2d;
        const Vector3d scaled2dPoint = Scale(identity2d, scaling2d) * point2d;
        EXPECT_TRUE(translated2dPoint.isApprox(Vector3d{5.0, 7.0, 1.0}));
        EXPECT_TRUE(scaled2dPoint.isApprox(Vector3d{8.0, 15.0, 1.0}));
    }

    // Scale: diagonal action on points; last column unchanged.
    {
        const Vector3d scaling{2.0, 3.0, 4.0};
        const Matrix4d scaled = Scale(identity, scaling);
        const Vector4d unitPoint{1.0, 1.0, 1.0, 1.0};
        const Matrix4d reflected = Scale(identity, Vector3d{1.0, -1.0, 1.0});
        const Vector4d point{1.0, 2.0, 3.0, 1.0};

        Matrix4d reference = Matrix4d::Identity();
        reference.diagonal().head(3) = scaling;
        EXPECT_TRUE(scaled.isApprox(reference));
        EXPECT_TRUE((scaled * unitPoint).isApprox(Vector4d{2.0, 3.0, 4.0, 1.0}));
        EXPECT_TRUE((reflected * point).isApprox(Vector4d{1.0, -2.0, 3.0, 1.0}));
    }

    // Composition is not commutative: Translate*Rotate vs Rotate*Translate.
    {
        const Vector3d translation{1.0, 0.0, 0.0};
        const Matrix4d translateThenRotate =
            Rotate(Translate(identity, translation), HalfPi<double>, zAxis);
        const Matrix4d rotateThenTranslate =
            Translate(Rotate(identity, HalfPi<double>, zAxis), translation);

        EXPECT_FALSE(translateThenRotate.isApprox(rotateThenTranslate));
        EXPECT_TRUE((translateThenRotate * origin).isApprox(Vector4d{1.0, 0.0, 0.0, 1.0}));
        EXPECT_TRUE((rotateThenTranslate * origin).isApprox(yPoint));
    }

    // glm-style model-view composition.
    {
        const double translate = -2.5;
        const Vector2d rotate{Radians(30.0), Radians(45.0)};
        const Vector3d viewTranslation{0.0, 0.0, translate};
        const Vector3d pitchAxis{-1.0, 0.0, 0.0};
        const Vector3d yawAxis{0.0, 1.0, 0.0};

        Matrix4d view = Translate(identity, viewTranslation);
        view = Rotate(view, rotate.y(), pitchAxis);
        view = Rotate(view, rotate.x(), yawAxis);
        const Matrix4d model = Scale(identity, Vector3d{0.5, 0.5, 0.5});
        const Matrix4d modelView = view * model;

        EXPECT_NEAR(modelView.col(0).head(3).norm(), 0.5, 1e-12);
        EXPECT_TRUE(modelView.col(3).isApprox(Vector4d{0.0, 0.0, translate, 1.0}));
    }

}
