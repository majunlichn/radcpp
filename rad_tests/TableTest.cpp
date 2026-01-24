#include <rad/IO/Table.h>
#include <rad/IO/Logging.h>

#include <random>

#include <gtest/gtest.h>

TEST(IO, Table)
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    constexpr size_t recordCount = 8;
    rad::Table table;
    table.Reserve(1 + 4, 5);
    // Header
    table.AddRow().AddCol("VertexID").AddCol("x").AddCol("y").AddCol("z").AddCol("w");
    // Records
    for (size_t i  = 0; i < recordCount; ++i)
    {
        table.AddRow().AddCol(i).AddCol(dist(rng)).AddCol(dist(rng)).AddCol(dist(rng)).AddCol(dist(rng));
    }
    table.Select(1, 4);
    for (size_t i = 0; i < recordCount; ++i)
    {
        table.SetValue(1.0).NextRow();
    }
    RAD_LOG(info, "TableFormatter: \n{}",
        rad::TableFormatter(table.ToStringTable())
        .SetColAlignment(rad::TableFormatter::ColAlignment::Right)
        .SetHeaderBorder('=')
        .SetBottomBorder('-')
        .SetColSeperator(',')
        .Format());
}
