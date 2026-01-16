#include <rad/IO/Format.h>

namespace rad
{

size_t TableFormatter::GetMaxColCount() const
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

void TableFormatter::ReserveRows(size_t numRows)
{
    m_rows.reserve(numRows);
    size_t maxColCount = GetMaxColCount();
    for (auto& row : m_rows)
    {
        row.reserve(maxColCount);
    }
    m_rowSeparators.reserve(numRows + 1);
}

void TableFormatter::ReserveCols(size_t numCols)
{
    for (auto& row : m_rows)
    {
        row.reserve(numCols);
    }
    m_colWidths.reserve(numCols);
    m_colAlignments.reserve(numCols);
    m_colSeparators.reserve(numCols + 1);
}

void TableFormatter::Reserve(size_t numRows, size_t numCols)
{
    ReserveRows(numRows);
    ReserveCols(numCols);
}

void TableFormatter::ResizeRows(size_t numRows)
{
    m_rows.resize(numRows);
    size_t maxColCount = GetMaxColCount();
    for (auto& row : m_rows)
    {
        row.resize(maxColCount);
    }
    m_rowSeparators.resize(numRows + 1);
}

void TableFormatter::ResizeCols(size_t numCols)
{
    for (auto& row : m_rows)
    {
        row.resize(numCols);
    }
    m_colWidths.resize(numCols);
    m_colAlignments.resize(numCols);
    m_colSeparators.resize(numCols + 1);
}

void TableFormatter::Resize(size_t numRows, size_t numCols)
{
    ResizeRows(numRows);
    ResizeCols(numCols);
}

void TableFormatter::Clear()
{
    m_rows.clear();
    m_currRowIndex = 0;
    m_currColIndex = 0;
    m_colWidths.clear();
    m_colAlignments.clear();
}

void TableFormatter::SetColAlignment(size_t colIndex, ColAlignment alignment)
{
    if (colIndex >= GetMaxColCount())
    {
        ResizeCols(colIndex + 1);
    }
    m_colAlignments[colIndex] = alignment;
}

void TableFormatter::SetAlignment(ColAlignment alignment)
{
    for (size_t colIndex = 0; colIndex < GetMaxColCount(); ++colIndex)
    {
        SetColAlignment(colIndex, alignment);
    }
}

std::string TableFormatter::Align(const std::string& formatted, const size_t colWidth, ColAlignment alignment)
{
    assert(formatted.size() <= colWidth);
    if (alignment == ColAlignment::Left)
    {
        return formatted + std::string(colWidth - formatted.size(), ' ');
    }
    else if (alignment == ColAlignment::Right)
    {
        return std::string(colWidth - formatted.size(), ' ') + formatted;
    }
    else //if (alignment == Alignment::Center)
    {
        size_t leftPadding = (colWidth - formatted.size()) / 2;
        size_t rightPadding = colWidth - formatted.size() - leftPadding;
        return std::string(leftPadding, ' ') + formatted + std::string(rightPadding, ' ');
    }
}

void TableFormatter::Format(size_t rowIndex, size_t colIndex)
{
    const auto& cell = m_rows[rowIndex][colIndex];
    switch (cell.type)
    {
    case CellType::String:
        m_rows[rowIndex][colIndex].formatted = std::vformat(m_format.stringFormat, std::make_format_args(std::get<std::string>(cell.value)));
        break;
    case CellType::Float:
    {
        double value = std::get<double>(cell.value);
        if (m_format.floatAdaptiveSci && (value != 0) &&
            ((std::abs(value) >= m_format.floatSciThreshold) || (std::abs(value) < 1.0 / m_format.floatSciThreshold)))
        {
            m_rows[rowIndex][colIndex].formatted = std::vformat("{:.4e}", std::make_format_args(value));
            break;
        }
        else
        {
            m_rows[rowIndex][colIndex].formatted = std::vformat("{:.4f}", std::make_format_args(value));
            break;
        }
        break;
    }
    case CellType::Int64:
        m_rows[rowIndex][colIndex].formatted = std::vformat(m_format.intFormat, std::make_format_args(std::get<int64_t>(cell.value)));
        break;
    case CellType::UInt64:
        m_rows[rowIndex][colIndex].formatted = std::vformat(m_format.uintFormat, std::make_format_args(std::get<uint64_t>(cell.value)));
        break;
    case CellType::Bool:
        m_rows[rowIndex][colIndex].formatted = std::get<bool>(cell.value) ? m_format.boolTrueString : m_format.boolFalseString;
        break;
    }
    if (m_rows[rowIndex][colIndex].formatted.size() > m_colWidths[colIndex])
    {
        m_colWidths[colIndex] = m_rows[rowIndex][colIndex].formatted.size();
    }
}

std::string RepeatFill(std::string_view pattern, size_t width)
{
    if (pattern.empty() || (width == 0))
    {
        return "";
    }
    std::string result;
    result.reserve(width);
    while (result.size() + pattern.size() <= width)
    {
        result += pattern;
    }
    result += pattern.substr(0, width - result.size());
    return result;
}

std::string TableFormatter::Print()
{
    for (size_t rowIndex = 0; rowIndex < m_rows.size(); ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < m_rows[rowIndex].size(); ++colIndex)
        {
            Format(rowIndex, colIndex);
        }
    }

    size_t maxColWidth = 0;
    for (const auto& width : m_colWidths)
    {
        if (maxColWidth < width)
        {
            maxColWidth = width;
        }
    }

    size_t tableWidth = m_colSeparators[0].size();
    for (size_t colIndex = 0; colIndex < GetColCount(0); ++colIndex)
    {
        size_t colWidth = m_unifiedColumnWidth ? maxColWidth : m_colWidths[colIndex];
        tableWidth += colWidth;
        tableWidth += m_colSeparators[colIndex + 1].size();
    }
    tableWidth += m_colMargin * (GetMaxColCount() - 1);

    for (size_t rowIndex = 0; rowIndex < m_rowSeparators.size(); ++rowIndex)
    {
        if (!m_rowSeparators[rowIndex].empty())
        {
            m_rowSeparators[rowIndex] = RepeatFill(m_rowSeparators[rowIndex], tableWidth);
        }
    }

    std::string buffer;
    buffer.reserve(4 * 1024 * 1024);
    if (!m_rowSeparators[0].empty())
    {
        buffer += m_rowSeparators[0] + '\n';
    }
    for (size_t rowIndex = 0; rowIndex < m_rows.size(); ++rowIndex)
    {
        buffer += m_colSeparators[0];
        for (size_t colIndex = 0; colIndex < m_rows[rowIndex].size(); ++colIndex)
        {
            const auto& cell = m_rows[rowIndex][colIndex];
            size_t colWidth = m_unifiedColumnWidth ? maxColWidth : m_colWidths[colIndex];
            colWidth = std::max<size_t>(colWidth, m_minColWidth);
            if (colIndex < m_rows[rowIndex].size() - 1)
            {
                colWidth += m_colMargin;
            }
            ColAlignment alignment = m_colAlignments[colIndex];
            buffer += Align(cell.formatted, colWidth, alignment) + m_colSeparators[colIndex + 1];
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
