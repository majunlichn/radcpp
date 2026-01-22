#pragma once

#include <rad/Common/Platform.h>
#include <rad/Common/Float.h>
#include <rad/Common/Integer.h>
#include <rad/Common/RefCounted.h>
#include <rad/Common/String.h>
#include <rad/Common/Flags.h>
#include <rad/Container/ArrayRef.h>
#include <rad/Container/Span.h>

#include <variant>

namespace rad
{

class StringTable
{
public:
    std::vector<std::vector<std::string>> m_rows;
    size_t m_selectedRowIndex = 0;
    size_t m_selectedColIndex = 0;

    StringTable() = default;
    ~StringTable() = default;
    size_t GetRowCount() const
    {
        return m_rows.size();
    }
    size_t GetColCount(size_t rowIndex) const
    {
        return m_rows[rowIndex].size();
    }
    size_t GetMaxColCount() const;
    void ReserveRows(size_t rowCount);
    void ReserveCols(size_t colCount);
    void Reserve(size_t rowCount, size_t colCount);
    void ResizeRows(size_t rowCount);
    void ResizeCols(size_t colCount);
    void Resize(size_t rowCount, size_t colCount);
    void Clear();

    StringTable& Select(size_t rowIndex, size_t colIndex)
    {
        m_selectedRowIndex = rowIndex;
        m_selectedColIndex = colIndex;
        return *this;
    }

    StringTable& NextRow()
    {
        m_selectedRowIndex += 1;
        return *this;
    }

    StringTable& NextCol()
    {
        m_selectedColIndex += 1;
        return *this;
    }

    StringTable& SetValue(size_t rowIndex, size_t colIndex, std::string_view value)
    {
        m_rows[rowIndex][colIndex] = value;
        return *this;
    }

    StringTable& SetValue(std::string_view value)
    {
        return SetValue(m_selectedRowIndex, m_selectedColIndex, value);
    }

    StringTable& AddRow()
    {
        m_rows.emplace_back();
        m_selectedRowIndex = m_rows.size() - 1;
        return *this;
    }

    template <typename T>
    StringTable& AddCol(const T& value)
    {
        m_rows[m_selectedRowIndex].emplace_back();
        m_selectedColIndex = m_rows[m_selectedRowIndex].size() - 1;
        SetValue(m_selectedRowIndex, m_selectedColIndex, value);
        return *this;
    }

}; // class StringTable

class Table : public RefCounted<Table>
{
public:
    enum class DataType
    {
        String,
        Float,
        Int64,
        UInt64,
        Bool,
    };

    struct Cell
    {
        DataType type = DataType::String;
        std::variant<double, int64_t, uint64_t, bool, std::string> value = "";
    }; // struct Cell

    std::vector<std::vector<Cell>> m_rows;
    size_t m_selectedRowIndex = 0;
    size_t m_selectedColIndex = 0;

    struct CellFormat
    {
        std::string stringFormat = "{}";
        std::string floatFormat = "{:.4f}";
        std::string intFormat = "{}";
        std::string uintFormat = "{}";
        std::string boolTrueString = "True";
        std::string boolFalseString = "False";
    };
    std::vector<CellFormat> m_colFormats;

    Table() = default;
    ~Table() = default;
    Table(size_t rowCount, size_t colCount)
    {
        Resize(rowCount, colCount);
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

    void ReserveRows(size_t rowCount);
    void ReserveCols(size_t colCount);
    void Reserve(size_t rowCount, size_t colCount);

    void ResizeRows(size_t rowCount);
    void ResizeCols(size_t colCount);
    void Resize(size_t rowCount, size_t colCount);

    void Clear();

    template <typename T>
    Table& SetValue(size_t rowIndex, size_t colIndex, T value);
    template <typename T>
    T GetValue(size_t rowIndex, size_t colIndex) const;

    Table& Select(size_t rowIndex, size_t colIndex)
    {
        m_selectedRowIndex = rowIndex;
        m_selectedColIndex = colIndex;
        return *this;
    }

    Table& NextRow()
    {
        m_selectedRowIndex += 1;
        return *this;
    }

    Table& NextCol()
    {
        m_selectedColIndex += 1;
        return *this;
    }

    template <typename T>
    Table& SetValue(T&& value)
    {
        return SetValue<T>(m_selectedRowIndex, m_selectedColIndex, std::forward<T>(value));
    }

    Table& AddRow()
    {
        m_rows.emplace_back();
        m_selectedRowIndex = m_rows.size() - 1;
        return *this;
    }

    template <typename T>
    Table& AddCol(const T& value)
    {
        m_rows[m_selectedRowIndex].emplace_back();
        m_selectedColIndex = m_rows[m_selectedRowIndex].size() - 1;
        SetValue(m_selectedRowIndex, m_selectedColIndex, value);
        return *this;
    }

    static std::string FormatCell(const Table::Cell& cell, const CellFormat& format);
    std::string FormatCell(size_t rowIndex, size_t colIndex);
    StringTable Format();

}; // class Table

template<typename T>
inline Table& Table::SetValue(size_t rowIndex, size_t colIndex, T value)
{
    if (rowIndex >= m_rows.size())
    {
        ResizeRows(rowIndex + 1);
    }
    if (colIndex >= m_rows[rowIndex].size())
    {
        m_rows[rowIndex].resize(rowIndex + 1);
    }

    if constexpr (std::is_same_v<T, std::string>)
    {
        m_rows[rowIndex][colIndex].type = DataType::String;
        m_rows[rowIndex][colIndex].value = std::move(value);
    }
    else if constexpr (std::is_same_v<T, const char*>)
    {
        m_rows[rowIndex][colIndex].type = DataType::String;
        m_rows[rowIndex][colIndex].value = value;
    }
    else if constexpr (is_floating_point_v<T>)
    {
        m_rows[rowIndex][colIndex].type = DataType::Float;
        m_rows[rowIndex][colIndex].value = double(value);
    }
    else if constexpr (is_signed_integer_v<T>)
    {
        m_rows[rowIndex][colIndex].type = DataType::Int64;
        m_rows[rowIndex][colIndex].value = int64_t(value);
    }
    else if constexpr (is_unsigned_integer_v<T>)
    {
        m_rows[rowIndex][colIndex].type = DataType::UInt64;
        m_rows[rowIndex][colIndex].value = uint64_t(value);
    }
    else if constexpr (std::is_same_v<T, bool>)
    {
        m_rows[rowIndex][colIndex].type = DataType::Bool;
        m_rows[rowIndex][colIndex].value = value;
    }
    else
    {
        assert(false && "invalid data type!");
        RAD_UNREACHABLE();
    }
    return *this;
}

template<typename T>
inline T Table::GetValue(size_t rowIndex, size_t colIndex) const
{
    assert(rowIndex < m_rows.size());
    assert(colIndex < m_rows[rowIndex].size());

    switch (m_rows[rowIndex][colIndex].type)
    {
    case DataType::String:
        return static_cast<T>(std::get<std::string>(m_rows[rowIndex][colIndex].value));
    case DataType::Float:
        return static_cast<T>(std::get<double>(m_rows[rowIndex][colIndex].value));
    case DataType::Int64:
        return static_cast<T>(std::get<int64_t>(m_rows[rowIndex][colIndex].value));
    case DataType::UInt64:
        return static_cast<T>(std::get<uint64_t>(m_rows[rowIndex][colIndex].value));
    case DataType::Bool:
        return static_cast<T>(std::get<bool>(m_rows[rowIndex][colIndex].value));
    }
    RAD_UNREACHABLE();
    return T();
}

class TableFormatter
{
public:
    const StringTable* m_formatted = nullptr;

    std::vector<size_t> m_colWidths;
    size_t m_minColWidth = 0;
    std::vector<size_t> m_colMargins;

    enum class ColAlignment
    {
        Left,
        Right,
        Center,
    };
    std::vector<ColAlignment> m_colAlignments;

    std::vector<std::string> m_rowSeparators;
    std::vector<std::string> m_colSeparators;

    TableFormatter();
    TableFormatter(const StringTable& table);
    ~TableFormatter();

    void SetTable(const StringTable& table);

    void SetColMargin(size_t colIndex, size_t colMargin);
    void SetColAlignment(size_t colIndex, ColAlignment alignment);
    void SetColAlignment(ColAlignment alignment);

    void SetLeftBorder(const char sep = '|')
    {
        m_colSeparators[0] = sep;
    }

    void SetLeftBorder(std::string_view sep)
    {
        m_colSeparators[0] = sep;
    }

    void SetRightBorder(const char sep = '|')
    {
        m_colSeparators.back() = sep;
    }

    void SetRightBorder(std::string_view sep)
    {
        m_colSeparators.back() = sep;
    }

    void SetTopBorder(const char sep = '-')
    {
        m_rowSeparators[0] = sep;
    }

    void SetBottomBorder(const char sep = '-')
    {
        m_rowSeparators.back() = sep;
    }

    void SetHeaderBorder(const char sep = '-')
    {
        if (m_rowSeparators.size() >= 2)
        {
            m_rowSeparators[0] = sep;
            m_rowSeparators[1] = sep;
        }
    }

    void SetColSeperator(const char sep = ',')
    {
        for (size_t colIndex = 1; colIndex < m_colSeparators.size() - 1; ++colIndex)
        {
            m_colSeparators[colIndex] = sep;
        }
    }

    static std::string Align(const std::string& str, const size_t colWidth, ColAlignment alignment);
    size_t GetMaxColWidth() const;
    size_t GetTableWidth() const;
    void NormalizeColWidths(size_t colIndexMin, size_t colIndexMax);
    void NormalizeColWidths();
    std::string Format();

}; // class TableFormatter

} // namespace rad
