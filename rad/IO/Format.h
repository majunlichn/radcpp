#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/String.h>
#include <rad/Common/Flags.h>
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
    size_t m_currRowIndex = 0;
    size_t m_currColIndex = 0;
    struct TextFormat
    {
        std::string floatFormat = "{:.4f}";
        bool floatAdaptiveSci = true;
        // if value != 0, or abs(value) >= threshold, or abs(value) < 1/threshold, use scientific format
        float floatSciThreshold = 1e+04f;
        std::string floatSciFormat = "{:.4e}";
        std::string intFormat = "{}";
        std::string uintFormat = "{}";
        std::string boolTrueString = "True";
        std::string boolFalseString = "False";
        std::string stringFormat = "{}";
    } m_format;

    enum class CellAlignment
    {
        Left,
        Right,
        Center,
    };

    std::vector<size_t> m_colWidths;
    std::vector<CellAlignment> m_colAlignments;

    TableFormatter() = default;
    ~TableFormatter() = default;
    TableFormatter(size_t numRows, size_t numCols)
    {
        Resize(numRows, numCols);
    }

    size_t GetRowCount() const
    {
        return m_rows.size();
    }

    size_t GetColCount(size_t rowIndex) const
    {
        return m_rows[rowIndex].size();
    }

    size_t GetMaxColCount() const;

    void Reserve(size_t numRows, size_t numCols)
    {
        m_rows.reserve(numRows);
        for (auto& row : m_rows)
        {
            row.reserve(numCols);
        }
        m_colWidths.reserve(numCols);
        m_colAlignments.reserve(numCols);
    }

    void ResizeRows(size_t numRows)
    {
        m_rows.resize(numRows);
    }

    void ResizeColumns(size_t numCols)
    {
        for (auto& row : m_rows)
        {
            row.resize(numCols);
        }
        m_colWidths.resize(numCols);
        m_colAlignments.resize(numCols);
    }

    void Resize(size_t numRows, size_t numCols)
    {
        ResizeRows(numRows);
        ResizeColumns(numCols);
    }

    void Clear()
    {
        m_rows.clear();
        m_currRowIndex = 0;
        m_currColIndex = 0;
        m_colWidths.clear();
        m_colAlignments.clear();
    }

    template <typename T>
    void SetValue(size_t rowIndex, size_t colIndex, const T& value);
    template <typename T>
    T GetValue(size_t rowIndex, size_t colIndex) const;

    void SetCurrentCell(size_t rowIndex, size_t colIndex)
    {
        m_currRowIndex = rowIndex;
        m_currColIndex = colIndex;
    }

    template <typename T>
    void AddCell(const T& value)
    {
        SetValue(m_currRowIndex, m_currColIndex, value);
        m_currColIndex++;
    }

    void NextRow()
    {
        m_currRowIndex++;
        m_currColIndex = 0;
    }

    static std::string Align(const std::string& formatted, const size_t colWidth, CellAlignment alignment);
    void Format(size_t rowIndex, size_t colIndex);

    struct PrintOptions
    {
        bool unifiedColumnWidth = false;
        char columnSeparator = ',';
    } m_printOptions;

    std::string Print(const PrintOptions& options = {});

}; // class TableFormatter

template<typename T>
inline void TableFormatter::SetValue(size_t rowIndex, size_t colIndex, const T& value)
{
    if (rowIndex >= m_rows.size())
    {
        ResizeRows(rowIndex + 1);
    }
    if (colIndex >= m_rows[rowIndex].size())
    {
        ResizeColumns(colIndex + 1);
    }

    if constexpr (is_floating_point_v<T>)
    {
        m_rows[rowIndex][colIndex].type = CellType::Float;
        m_rows[rowIndex][colIndex].value = double(value);
    }
    else if constexpr (is_signed_integer_v<T>)
    {
        m_rows[rowIndex][colIndex].type = CellType::Int64;
        m_rows[rowIndex][colIndex].value = int64_t(value);
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        m_rows[rowIndex][colIndex].type = CellType::Bool;
        m_rows[rowIndex][colIndex].value = bool(value);
    }
    else if constexpr (is_unsigned_integer_v<T>)
    {
        m_rows[rowIndex][colIndex].type = CellType::UInt64;
        m_rows[rowIndex][colIndex].value = uint64_t(value);
    }
    else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, const char*>)
    {
        m_rows[rowIndex][colIndex].type = CellType::String;
        m_rows[rowIndex][colIndex].value = value;
    }
    else
    {
        RAD_UNREACHABLE();
        return;
    }
}

template<typename T>
inline T TableFormatter::GetValue(size_t rowIndex, size_t colIndex) const
{
    assert(rowIndex < m_rows.size());
    assert(colIndex < m_rows[rowIndex].size());

    switch (m_rows[rowIndex][colIndex].type)
    {
    case CellType::Float:
        return static_cast<T>(std::get<double>(m_rows[rowIndex][colIndex].value));
    case CellType::Int64:
        return static_cast<T>(std::get<int64_t>(m_rows[rowIndex][colIndex].value));
    case CellType::UInt64:
        return static_cast<T>(std::get<uint64_t>(m_rows[rowIndex][colIndex].value));
    case CellType::Bool:
        return static_cast<T>(std::get<bool>(m_rows[rowIndex][colIndex].value));
    case CellType::String:
        return static_cast<T>(std::get<std::string>(m_rows[rowIndex][colIndex].value));
    }
    RAD_UNREACHABLE();
    return T();
}

} // namespace rad
