#include <rad/IO/Format.h>
#include <rad/IO/Logging.h>

#include <gtest/gtest.h>

TEST(IO, TableFormatter)
{
    const size_t numRows = 4;
    const size_t numCols = 4;
    rad::TableFormatter table(numRows, numCols);
    for (size_t row  = 0; row < numRows; ++row)
    {
        table.SetValue<double>(row, 0, 1.2345 + row);
        table.SetValue<int64_t>(row, 1, 100 + row);
        table.SetValue<uint64_t>(row, 2, 200 + row);
        table.SetValue<bool>(row, 3, (row % 2) == 0);
    }
    RAD_LOG(info, "TableFormatter: \n{}", table.Print());
}
