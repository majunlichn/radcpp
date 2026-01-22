#include <rad/IO/Table.h>

namespace rad
{

size_t StringTable::GetMaxColCount() const
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

void StringTable::ReserveRows(size_t rowCount)
{
    m_rows.reserve(rowCount);
}

void StringTable::ReserveCols(size_t colCount)
{
    for (auto& row : m_rows)
    {
        row.reserve(colCount);
    }
}

void StringTable::Reserve(size_t rowCount, size_t colCount)
{
    ReserveRows(rowCount);
    ReserveCols(colCount);
}

void StringTable::ResizeRows(size_t rowCount)
{
    m_rows.resize(rowCount);
}

void StringTable::ResizeCols(size_t colCount)
{
    for (auto& row : m_rows)
    {
        row.resize(colCount);
    }
}

void StringTable::Resize(size_t rowCount, size_t colCount)
{
    ResizeRows(rowCount);
    ResizeCols(colCount);
}

void StringTable::Clear()
{
    m_rows.clear();
    m_selectedRowIndex = 0;
    m_selectedColIndex = 0;
}

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

std::string Table::FormatCell(const Table::Cell& cell, const CellFormat& format)
{
    switch (cell.type)
    {
    case Table::DataType::String:
        return std::vformat(format.stringFormat, std::make_format_args(std::get<std::string>(cell.value)));
        break;
    case Table::DataType::Float:
        return  std::vformat(format.floatFormat, std::make_format_args(std::get<double>(cell.value)));
        break;
    case Table::DataType::Int64:
        return std::vformat(format.intFormat, std::make_format_args(std::get<int64_t>(cell.value)));
        break;
    case Table::DataType::UInt64:
        return std::vformat(format.uintFormat, std::make_format_args(std::get<uint64_t>(cell.value)));
        break;
    case Table::DataType::Bool:
        return std::get<bool>(cell.value) ? format.boolTrueString : format.boolFalseString;
        break;
    }
    RAD_UNREACHABLE();
    return {};
}

std::string Table::FormatCell(size_t rowIndex, size_t colIndex)
{
    return FormatCell(m_rows[rowIndex][colIndex], m_colFormats[colIndex]);
}

StringTable Table::Format()
{
    m_colFormats.resize(GetMaxColCount());
    StringTable formatted;
    formatted.ResizeRows(GetRowCount());
    for (size_t rowIndex = 0; rowIndex < GetRowCount(); ++rowIndex)
    {
        formatted.m_rows[rowIndex].resize(GetColCount(rowIndex));
        for (size_t colIndex = 0; colIndex < GetColCount(rowIndex); ++colIndex)
        {
            formatted.m_rows[rowIndex][colIndex] = FormatCell(m_rows[rowIndex][colIndex], m_colFormats[colIndex]);
        }
    }
    return formatted;
}

TableFormatter::TableFormatter()
{
}

TableFormatter::TableFormatter(const StringTable& table)
{
    SetTable(table);
}

TableFormatter::~TableFormatter()
{
}

void TableFormatter::SetTable(const StringTable& table)
{
    m_formatted = &table;

    size_t rowCount = m_formatted->GetRowCount();
    size_t maxColCount = m_formatted->GetMaxColCount();

    m_colWidths.resize(maxColCount, 0);
    m_colMargins.resize(maxColCount, 1);
    m_colAlignments.resize(maxColCount, ColAlignment::Left);
    m_rowSeparators.resize(rowCount + 1);
    m_colSeparators.resize(maxColCount + 1, ",");
    m_colSeparators[0].clear();
    m_colSeparators.back().clear();

    for (size_t rowIndex = 0; rowIndex < rowCount; ++rowIndex)
    {
        size_t colCount = m_formatted->GetColCount(rowIndex);
        for (size_t colIndex = 0; colIndex < colCount; ++colIndex)
        {
            const auto& cell = m_formatted->m_rows[rowIndex][colIndex];
            if (m_colWidths[colIndex] < cell.size())
            {
                m_colWidths[colIndex] = cell.size();
            }
        }
    }

    // Apply colMargin and minColWidth
    for (size_t colIndex = 0; colIndex < m_colWidths.size(); ++colIndex)
    {
        m_colWidths[colIndex] += m_colMargins[colIndex];
        if (m_colAlignments[colIndex] == ColAlignment::Center)
        {
            m_colWidths[colIndex] += m_colMargins[colIndex];
        }
        m_colWidths[colIndex] = std::max<size_t>(m_colWidths[colIndex], m_minColWidth);
    }
}

void TableFormatter::SetColMargin(size_t colIndex, size_t colMargin)
{
    if (m_colMargins.size() <= colIndex)
    {
        m_colMargins.resize(colIndex + 1, 1);
    }
    m_colMargins[colIndex] = colMargin;
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

size_t TableFormatter::GetMaxColWidth() const
{
    size_t maxColWidth = 0;
    for (const auto& width : m_colWidths)
    {
        if (maxColWidth < width)
        {
            maxColWidth = width;
        }
    }
    return maxColWidth;
}

size_t TableFormatter::GetTableWidth() const
{
    size_t tableWidth = m_colSeparators[0].size();
    for (size_t colIndex = 0; colIndex < m_colWidths.size(); ++colIndex)
    {
        tableWidth += m_colWidths[colIndex];
        tableWidth += m_colSeparators[colIndex + 1].size();
    }
    return tableWidth;
}

void TableFormatter::NormalizeColWidths(size_t colIndexMin, size_t colIndexMax)
{
    assert(colIndexMin < colIndexMax);
    size_t maxColWidth = 0;
    for (size_t colIndex = colIndexMin; colIndex < colIndexMax; ++colIndex)
    {
        if (maxColWidth < m_colWidths[colIndex])
        {
            maxColWidth = m_colWidths[colIndex];
        }
    }
    for (size_t colIndex = colIndexMin; colIndex < colIndexMax; ++colIndex)
    {
        m_colWidths[colIndex] = maxColWidth;
    }
}

void TableFormatter::NormalizeColWidths()
{
    size_t maxColWidth = GetMaxColWidth();
    for (auto& colWidth : m_colWidths)
    {
        colWidth = maxColWidth;
    }
}

std::string TableFormatter::Format()
{
    const auto& rows = m_formatted->m_rows;
    size_t maxColWidth = GetMaxColWidth();
    size_t tableWidth = GetTableWidth();

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
            size_t colWidth = m_colWidths[colIndex];
            ColAlignment alignment = m_colAlignments[colIndex];
            buffer += Align(m_formatted->m_rows[rowIndex][colIndex], colWidth, alignment) + m_colSeparators[colIndex + 1];
        }
        buffer += '\n';
        if (!m_rowSeparators[rowIndex + 1].empty())
        {
            buffer += m_rowSeparators[rowIndex + 1] + '\n';
        }
    }
    return buffer;
}

} // namespace rad
