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
        String,
        Float,
        Int64,
        UInt64,
        Bool,
    }; // enum class CellType

    struct Cell
    {
        CellType type = CellType::String;
        std::variant<double, int64_t, uint64_t, bool, std::string> value = "";
        std::string formatted;
    }; // struct Cell

    std::vector<std::vector<Cell>> m_rows;
    size_t m_currRowIndex = 0;
    size_t m_currColIndex = 0;
    std::vector<size_t> m_colWidths;
    size_t m_minColWidth = 0;
    size_t m_colMargin = 1;
    bool m_unifiedColumnWidth = false;
    enum class ColAlignment
    {
        Left,
        Right,
        Center,
    };
    std::vector<ColAlignment> m_colAlignments;
    std::vector<std::string> m_rowSeparators;
    std::vector<std::string> m_colSeparators;

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

    void ReserveRows(size_t numRows);
    void ReserveCols(size_t numCols);
    void Reserve(size_t numRows, size_t numCols);

    void ResizeRows(size_t numRows);
    void ResizeCols(size_t numCols);
    void Resize(size_t numRows, size_t numCols);

    void Clear();

    template <typename T>
    void SetValue(size_t rowIndex, size_t colIndex, const T& value);
    template <typename T>
    T GetValue(size_t rowIndex, size_t colIndex) const;

    void Select(size_t rowIndex, size_t colIndex)
    {
        m_currRowIndex = rowIndex;
        m_currColIndex = colIndex;
    }

    void NextRow()
    {
        m_currRowIndex += 1;
        m_currColIndex = 0;
    }

    void NextCol()
    {
        m_currColIndex += 1;
    }

    template <typename T>
    void SetValue(const T& value)
    {
        SetValue(m_currRowIndex, m_currColIndex, value);
    }

    void SetColAlignment(size_t colIndex, ColAlignment alignment);
    void SetAlignment(ColAlignment alignment);

    void SetLeftBorder(const char border = '|')
    {
        m_colSeparators[0] = border;
    }

    void SetRightBorder(const char border = '|')
    {
        m_colSeparators.back() = border;
    }

    void SetTopBorder(const char border = '-')
    {
        m_rowSeparators[0] = border;
    }

    void SetBottomBorder(const char border = '-')
    {
        m_rowSeparators.back() = border;
    }

    void SetHeaderBorder(const char border = '-')
    {
        if (m_rows.size() >= 1)
        {
            m_rowSeparators[0] = border;
            m_rowSeparators[1] = border;
        }
    }

    void SetColInnerBorder(const char border = ',')
    {
        for (size_t colIndex = 1; colIndex < m_colSeparators.size() - 1; ++colIndex)
        {
            m_colSeparators[colIndex] = border;
        }
    }

    static std::string Align(const std::string& formatted, const size_t colWidth, ColAlignment alignment);
    void Format(size_t rowIndex, size_t colIndex);

    std::string Print();

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
        ResizeCols(colIndex + 1);
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
    else
    {
        m_rows[rowIndex][colIndex].type = CellType::String;
        m_rows[rowIndex][colIndex].value = std::string(value);
    }
}

template<typename T>
inline T TableFormatter::GetValue(size_t rowIndex, size_t colIndex) const
{
    assert(rowIndex < m_rows.size());
    assert(colIndex < m_rows[rowIndex].size());

    switch (m_rows[rowIndex][colIndex].type)
    {
    case CellType::String:
        return static_cast<T>(std::get<std::string>(m_rows[rowIndex][colIndex].value));
    case CellType::Float:
        return static_cast<T>(std::get<double>(m_rows[rowIndex][colIndex].value));
    case CellType::Int64:
        return static_cast<T>(std::get<int64_t>(m_rows[rowIndex][colIndex].value));
    case CellType::UInt64:
        return static_cast<T>(std::get<uint64_t>(m_rows[rowIndex][colIndex].value));
    case CellType::Bool:
        return static_cast<T>(std::get<bool>(m_rows[rowIndex][colIndex].value));
    }
    RAD_UNREACHABLE();
    return T();
}

} // namespace rad
