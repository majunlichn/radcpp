#include <rad/IO/Table.h>

namespace rad
{

size_t Table::GetMaxColCount() const
{
    size_t maxColCount = 0;
    for (const auto& row : m_rows)
    {
        if (maxColCount < row.size())
        {
            maxColCount = row.size();
        }
    }
    return maxColCount;
}

void Table::ReserveRows(size_t rowCount)
{
    m_rows.reserve(rowCount);
}

void Table::ReserveCols(size_t colCount)
{
    for (auto& row : m_rows)
    {
        row.reserve(colCount);
    }
}

void Table::Reserve(size_t rowCount, size_t colCount)
{
    ReserveRows(rowCount);
    ReserveCols(colCount);
}

void Table::ResizeRows(size_t rowCount)
{
    m_rows.resize(rowCount);
}

void Table::ResizeCols(size_t colCount)
{
    for (auto& row : m_rows)
    {
        row.resize(colCount);
    }
}

void Table::Resize(size_t rowCount, size_t colCount)
{
    ResizeRows(rowCount);
    ResizeCols(colCount);
}

void Table::Clear()
{
    m_rows.clear();
    m_selectedRowIndex = 0;
    m_selectedColIndex = 0;
}

TableFormatter::TableFormatter()
{
}

TableFormatter::TableFormatter(const Table& table)
{
    SetTable(table);
}

TableFormatter::~TableFormatter()
{
}

void TableFormatter::SetTable(const Table& table)
{
    m_table = &table;
    m_colMargins.resize(m_table->GetMaxColCount(), 1);
    m_colAlignments.resize(m_table->GetMaxColCount(), ColAlignment::Left);
    m_rowSeparators.resize(m_table->GetRowCount() + 1);
    m_colSeparators.resize(m_table->GetMaxColCount() + 1, ",");
    m_colSeparators[0].clear();
    m_colSeparators.back().clear();
}

void TableFormatter::SetColMargin(size_t colMargin)
{
}

void TableFormatter::SetColAlignment(size_t colIndex, ColAlignment alignment)
{
    m_colAlignments[colIndex] = alignment;
}

void TableFormatter::SetColAlignment(ColAlignment alignment)
{
    for (size_t colIndex = 0; colIndex < m_colAlignments.size(); ++colIndex)
    {
        SetColAlignment(colIndex, alignment);
    }
}

std::string TableFormatter::Align(const std::string& values, const size_t colWidth, ColAlignment alignment)
{
    if (colWidth <= values.size())
    {
        return values;
    }
    if (alignment == ColAlignment::Left)
    {
        return values + std::string(colWidth - values.size(), ' ');
    }
    else if (alignment == ColAlignment::Right)
    {
        return std::string(colWidth - values.size(), ' ') + values;
    }
    else //if (alignment == Alignment::Center)
    {
        size_t leftPadding = (colWidth - values.size()) / 2;
        size_t rightPadding = colWidth - values.size() - leftPadding;
        return std::string(leftPadding, ' ') + values + std::string(rightPadding, ' ');
    }
}

std::string TableFormatter::FormatValue(size_t rowIndex, size_t colIndex) const
{
    const auto& rows = m_table->m_rows;
    const auto& cell = rows[rowIndex][colIndex];
    switch (cell.type)
    {
    case Table::DataType::String:
        return std::vformat(m_format.stringFormat, std::make_format_args(std::get<std::string>(cell.value)));
        break;
    case Table::DataType::Float:
        return  std::vformat(m_format.floatFormat, std::make_format_args(std::get<double>(cell.value)));
        break;
    case Table::DataType::Int64:
        return std::vformat(m_format.intFormat, std::make_format_args(std::get<int64_t>(cell.value)));
        break;
    case Table::DataType::UInt64:
        return std::vformat(m_format.uintFormat, std::make_format_args(std::get<uint64_t>(cell.value)));
        break;
    case Table::DataType::Bool:
        return std::get<bool>(cell.value) ? m_format.boolTrueString : m_format.boolFalseString;
        break;
    }
    RAD_UNREACHABLE();
    return {};
}

std::string TableFormatter::FormatRow(size_t rowIndex, ArrayRef<size_t> colWidths) const
{
    std::string buffer;
    std::vector<std::string> values(m_table->GetColCount(rowIndex));
    for (size_t colIndex = 0; colIndex < m_table->GetColCount(rowIndex); ++colIndex)
    {
        values[colIndex] = FormatValue(rowIndex, colIndex);
    }
    const auto& rows = m_table->m_rows;
    buffer += m_colSeparators[0];
    for (size_t colIndex = 0; colIndex < rows[rowIndex].size(); ++colIndex)
    {
        const auto& cell = rows[rowIndex][colIndex];
        size_t colWidth = colWidths[colIndex];
        ColAlignment alignment = m_colAlignments[colIndex];
        buffer += Align(values[colIndex], colWidth, alignment) + m_colSeparators[colIndex + 1];
    }
    return buffer;
}

std::vector<std::vector<std::string>> TableFormatter::FormatValues(std::vector<size_t>& colWidths) const
{
    std::vector<std::vector<std::string>> values;
    values.resize(m_table->GetRowCount());
    for (size_t rowIndex = 0; rowIndex < m_table->GetRowCount(); ++rowIndex)
    {
        values[rowIndex].resize(m_table->GetColCount(rowIndex));
    }
    colWidths.resize(m_table->GetMaxColCount(), 0);
    for (size_t rowIndex = 0; rowIndex < m_table->GetRowCount(); ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < m_table->GetColCount(rowIndex); ++colIndex)
        {
            values[rowIndex][colIndex] = FormatValue(rowIndex, colIndex);
            size_t colWidth = values[rowIndex][colIndex].size();
            if (colWidths[colIndex] < colWidth)
            {
                colWidths[colIndex] = colWidth;
            }
        }
    }
    // Apply colMargin and minColWidth
    for (size_t colIndex = 0; colIndex < colWidths.size(); ++colIndex)
    {
        colWidths[colIndex] += m_colMargins[colIndex];
        if (m_colAlignments[colIndex] == ColAlignment::Center)
        {
            colWidths[colIndex] += m_colMargins[colIndex];
        }
        colWidths[colIndex] = std::max<size_t>(colWidths[colIndex], m_minColWidth);
    }
    return values;
}

size_t TableFormatter::GetMaxColWidth(ArrayRef<size_t> colWidths) const
{
    size_t maxColWidth = 0;
    for (const auto& width : colWidths)
    {
        if (maxColWidth < width)
        {
            maxColWidth = width;
        }
    }
    return maxColWidth;
}

size_t TableFormatter::GetTableWidth(ArrayRef<size_t> colWidths) const
{
    size_t tableWidth = m_colSeparators[0].size();
    for (size_t colIndex = 0; colIndex < m_table->GetMaxColCount(); ++colIndex)
    {
        tableWidth += colWidths[colIndex];
        tableWidth += m_colSeparators[colIndex + 1].size();
    }
    return tableWidth;
}

std::string TableFormatter::Format(const std::vector<std::vector<std::string>>& values, ArrayRef<size_t> colWidths)
{
    const auto& rows = m_table->m_rows;
    size_t maxColWidth = GetMaxColWidth(colWidths);

    size_t tableWidth = GetTableWidth(colWidths);

    for (size_t rowIndex = 0; rowIndex < m_rowSeparators.size(); ++rowIndex)
    {
        if (!m_rowSeparators[rowIndex].empty())
        {
            m_rowSeparators[rowIndex] = std::string(tableWidth, m_rowSeparators[rowIndex][0]);
        }
    }

    std::string buffer;
    buffer.reserve(4 * 1024 * 1024);
    if (!m_rowSeparators[0].empty())
    {
        buffer += m_rowSeparators[0] + '\n';
    }
    for (size_t rowIndex = 0; rowIndex < rows.size(); ++rowIndex)
    {
        buffer += m_colSeparators[0];
        for (size_t colIndex = 0; colIndex < rows[rowIndex].size(); ++colIndex)
        {
            const auto& cell = rows[rowIndex][colIndex];
            size_t colWidth = colWidths[colIndex];
            ColAlignment alignment = m_colAlignments[colIndex];
            buffer += Align(values[rowIndex][colIndex], colWidth, alignment) + m_colSeparators[colIndex + 1];
        }
        buffer += '\n';
        if (!m_rowSeparators[rowIndex + 1].empty())
        {
            buffer += m_rowSeparators[rowIndex + 1] + '\n';
        }
    }
    return buffer;
}

std::string TableFormatter::Format()
{
    const auto& rows = m_table->m_rows;
    std::vector<size_t> colWidths(m_table->GetMaxColCount(), 0);
    std::vector<std::vector<std::string>> values = FormatValues(colWidths);
    if (m_normalizeColWidth)
    {
        size_t maxColWidth = GetMaxColWidth(colWidths);
        for (auto& colWidth : colWidths)
        {
            colWidth = maxColWidth;
        }
    }
    return Format(values, colWidths);
}

} // namespace rad
