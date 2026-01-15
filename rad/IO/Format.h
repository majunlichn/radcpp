#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/String.h>
#include <rad/Container/Span.h>

#include <variant>

namespace rad
{

class TableFormatter
{
public:
    enum class CellType
    {
        Float,
        Int64,
        UInt64,
        Bool,
        String,
    }; // enum class CellType

    struct Cell
    {
        CellType type;
        std::variant<double, int64_t, uint64_t, bool, std::string> value;
        std::string formatted;
    };

    std::vector<std::vector<Cell>> m_rows;
    struct TextFormat
    {
        std::string floatFormat = "{:.6f}";
        std::string intFormat = "{}";
        std::string uintFormat = "{}";
        std::string boolTrueString = "True";
        std::string boolFalseString = "False";
        std::string stringFormat = "{}";
    } m_format;

    enum class Alignment
    {
        Left,
        Right,
        Center,
    };

    std::vector<size_t> m_colWidths;
    std::vector<Alignment> m_colAlignments;

    TableFormatter() = default;
    ~TableFormatter() = default;
    TableFormatter(size_t numRows, size_t numCols)
    {
        Resize(numRows, numCols);
    }

    void Resize(size_t numRows, size_t numCols)
    {
        m_rows.resize(numRows);
        for (auto& row : m_rows)
        {
            row.resize(numCols);
        }
        m_colWidths.resize(numCols);
        m_colAlignments.resize(numCols);
    }

    template <typename T>
    void Set(size_t row, size_t col, const T& value)
    {
        if constexpr (is_floating_point_v<T>)
        {
            m_rows[row][col].type = CellType::Float;
            m_rows[row][col].value = double(value);
        }
        else if constexpr (is_signed_integer_v<T>)
        {
            m_rows[row][col].type = CellType::Int64;
            m_rows[row][col].value = int64_t(value);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            m_rows[row][col].type = CellType::Bool;
            m_rows[row][col].value = bool(value);
        }
        else if constexpr (is_unsigned_integer_v<T>)
        {
            m_rows[row][col].type = CellType::UInt64;
            m_rows[row][col].value = uint64_t(value);
        }
        else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, const char*>)
        {
            m_rows[row][col].type = CellType::String;
            m_rows[row][col].value = value;
        }
        else
        {
            RAD_UNREACHABLE();
        }
    }

    template <typename T>
    T Get(size_t row, size_t col) const
    {
        assert(row < m_rows.size());
        assert(col < m_rows[row].size());

        switch (m_rows[row][col].type)
        {
        case CellType::Float:
            return static_cast<T>(std::get<double>(m_rows[row][col].value));
        case CellType::Int64:
            return static_cast<T>(std::get<int64_t>(m_rows[row][col].value));
        case CellType::UInt64:
            return static_cast<T>(std::get<uint64_t>(m_rows[row][col].value));
        case CellType::Bool:
            return static_cast<T>(std::get<bool>(m_rows[row][col].value));
        case CellType::String:
            return static_cast<T>(std::get<std::string>(m_rows[row][col].value));
        }
        RAD_UNREACHABLE();
    }

    std::string Align(const std::string& formatted, const size_t colWidth, Alignment alignment);
    std::string Print();

}; // class TableFormatter

} // namespace rad
