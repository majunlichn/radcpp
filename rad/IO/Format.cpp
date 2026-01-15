#include <rad/IO/Format.h>

namespace rad
{

std::string TableFormatter::Align(const std::string& formatted, const size_t colWidth, Alignment alignment)
{
    assert(formatted.size() <= colWidth);
    if (alignment == Alignment::Left)
    {
        return formatted + std::string(colWidth - formatted.size(), ' ');
    }
    else if (alignment == Alignment::Right)
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

std::string TableFormatter::Print()
{
    for (size_t row = 0; row < m_rows.size(); ++row)
    {
        for (size_t col = 0; col < m_rows[row].size(); ++col)
        {
            const auto& cell = m_rows[row][col];
            switch (cell.type)
            {
            case CellType::Float:
                m_rows[row][col].formatted = std::vformat(m_format.floatFormat, std::make_format_args(std::get<double>(cell.value)));
                break;
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

    std::string buffer;
    buffer.reserve(4 * 1024 * 1024);
    for (size_t row = 0; row < m_rows.size(); ++row)
    {
        for (size_t col = 0; col < m_rows[row].size(); ++col)
        {
            const auto& cell = m_rows[row][col];
            buffer += Align(cell.formatted, m_colWidths[col], m_colAlignments[col]) + ',';
        }
        buffer += '\n';
    }
    return buffer;
}

} // namespace rad
