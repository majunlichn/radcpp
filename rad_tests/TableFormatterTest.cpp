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

    table.SetValue("ID");
    table.NextCol();
    table.SetValue("A");
    table.NextCol();
    table.SetValue("B");
    table.NextCol();
    table.SetValue("C");
    table.NextCol();
    table.SetValue("D");
    table.NextRow();

    for (size_t row  = 0; row < 8; ++row)
    {
        table.SetValue(row);
        table.NextCol();
        table.SetValue(floatDist(rng));
        table.NextCol();
        table.SetValue(intDist(rng));
        table.NextCol();
        table.SetValue(uintDist(rng));
        table.NextCol();
        table.SetValue((row % 4) == 0);
        table.NextRow();
    }
    table.SetAlignment(rad::TableFormatter::ColAlignment::Right);
    table.SetHeaderBorder();
    table.SetBottomBorder();
    table.SetColInnerBorder(',');
    table.m_unifiedColumnWidth = true;
    RAD_LOG(info, "TableFormatter: \n{}", table.Print());
}
