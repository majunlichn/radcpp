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
        table.Set<double>(row, 0, 1.2345 + row);
        table.Set<int64_t>(row, 1, 100 + row);
        table.Set<uint64_t>(row, 2, 200 + row);
        table.Set<bool>(row, 3, (row % 2) == 0);
    }
    RAD_LOG(info, "TableFormatter: \n{}", table.Print());
}
