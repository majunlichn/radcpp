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

std::string TableFormatter::Align(const std::string& formatted, const size_t colWidth, CellAlignment alignment)
{
    assert(formatted.size() <= colWidth);
    if (alignment == CellAlignment::Left)
    {
        return formatted + std::string(colWidth - formatted.size(), ' ');
    }
    else if (alignment == CellAlignment::Right)
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

std::string TableFormatter::Print(const PrintOptions& options)
{
    for (size_t row = 0; row < m_rows.size(); ++row)
    {
        for (size_t col = 0; col < m_rows[row].size(); ++col)
        {
            const auto& cell = m_rows[row][col];
            switch (cell.type)
            {
            case CellType::Float:
            {
                double value = std::get<double>(cell.value);
                if (m_format.floatAdaptiveSci && (value != 0) &&
                    ((std::abs(value) >= m_format.floatSciThreshold) || (std::abs(value) < 1.0 / m_format.floatSciThreshold)))
                {
                    m_rows[row][col].formatted = std::vformat("{:.4e}", std::make_format_args(value));
                    break;
                }
                else
                {
                    m_rows[row][col].formatted = std::vformat("{:.4f}", std::make_format_args(value));
                    break;
                }
                break;
            }
            case CellType::Int64:
                m_rows[row][col].formatted = std::vformat(m_format.intFormat, std::make_format_args(std::get<int64_t>(cell.value)));
                break;
            case CellType::UInt64:
                m_rows[row][col].formatted = std::vformat(m_format.uintFormat, std::make_format_args(std::get<uint64_t>(cell.value)));
                break;
            case CellType::Bool:
                m_rows[row][col].formatted = std::get<bool>(cell.value) ? m_format.boolTrueString : m_format.boolFalseString;
                break;
            case CellType::String:
                m_rows[row][col].formatted = std::vformat(m_format.stringFormat, std::make_format_args(std::get<std::string>(cell.value)));
                break;
            }
            if (m_rows[row][col].formatted.size() > m_colWidths[col])
            {
                m_colWidths[col] = m_rows[row][col].formatted.size();
            }
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

    std::string buffer;
    buffer.reserve(4 * 1024 * 1024);
    for (size_t row = 0; row < m_rows.size(); ++row)
    {
        for (size_t col = 0; col < m_rows[row].size(); ++col)
        {
            const auto& cell = m_rows[row][col];
            size_t colWidth = options.unifiedColumnWidth ? maxColWidth : m_colWidths[col];
            CellAlignment alignment = m_colAlignments[col];
            buffer += Align(cell.formatted, colWidth, alignment) + options.columnSeparator;
        }
        buffer += '\n';
    }
    return buffer;
}

} // namespace rad
