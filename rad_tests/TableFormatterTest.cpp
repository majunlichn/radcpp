#include <rad/IO/Format.h>
#include <rad/IO/Logging.h>

#include <random>

#include <gtest/gtest.h>

TEST(IO, TableFormatter)
{
    std::default_random_engine rng;
    std::uniform_real_distribution<double> floatDist(0.0, 1000.0);
    std::uniform_int_distribution<int64_t> intDist(0, 1000);
    std::uniform_int_distribution<uint64_t> uintDist(0, 1000);

    rad::TableFormatter table;

    table.AddCol("ID");
    table.AddCol("A");
    table.AddCol("B");
    table.AddCol("C");
    table.AddCol("D");
    table.NextRow();

    for (size_t row  = 0; row < 8; ++row)
    {
        table.AddCol(row);
        table.AddCol(floatDist(rng));
        table.AddCol(intDist(rng));
        table.AddCol(uintDist(rng));
        table.AddCol((row % 4) == 0);
        table.NextRow();
    }
    table.SetAlignment(rad::TableFormatter::ColAlignment::Right);
    table.SetHeaderBorder();
    table.SetBottomBorder();
    table.SetColInnerBorder(',');
    RAD_LOG(info, "TableFormatter: \n{}", table.Print());
}
