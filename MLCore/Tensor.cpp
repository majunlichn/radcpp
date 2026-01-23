#include <MLCore/Tensor.h>
#include <MLCore/Backend.h>
#include <MLCore/Device.h>
#include <MLCore/Context.h>
#include <MLCore/TensorIterator.h>
#include <MLCore/Global.h>

#include <rad/IO/Table.h>

namespace ML
{

size_t GetTensorElementCount(rad::ArrayRef<size_t> sizes)
{
    if (sizes.empty())
    {
        return 0;
    }
    size_t count = sizes[0];
    for (size_t i = 1; i < sizes.size(); ++i)
    {
        count *= sizes[i];
    }
    return count;
}

std::vector<size_t> MakeTensorStrides(rad::ArrayRef<size_t> sizes, rad::ArrayRef<size_t> memoryOrder)
{
    assert(memoryOrder.empty() || (memoryOrder.size() == sizes.size()));

    std::vector<size_t> strides(sizes.size(), 0);
    if (memoryOrder.empty())
    {
        strides.back() = 1;
        std::partial_sum(
            sizes.rbegin(), sizes.rend() - 1, strides.rbegin() + 1, std::multiplies<size_t>());
    }
    else
    {
        size_t stride = 1;
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            strides[memoryOrder[i]] = stride;
            stride *= sizes[memoryOrder[i]];
        }
    }
    return strides;
}

size_t GetTensorDataSizeInElement(rad::ArrayRef<size_t> size, rad::ArrayRef<size_t> strides)
{
    size_t indexOfTheLastElement = 0;
    for (size_t i = 0; i < size.size(); ++i)
    {
        indexOfTheLastElement += (size[i] - 1) * strides[i];
    }
    return (indexOfTheLastElement + 1);
}

TensorStorage::TensorStorage(rad::Ref<Device> device) :
    m_device(std::move(device))
{
}

TensorStorage::~TensorStorage()
{
}

Tensor::Tensor()
{
}

Tensor::Tensor(rad::Ref<TensorStorage> storage, rad::Ref<Context> context) :
    m_storage(std::move(storage)),
    m_context(std::move(context))
{
    assert(m_storage->m_device == m_context->m_device);
    m_sizes = m_storage->m_sizes;
    m_strides = m_storage->m_strides;
    m_dataType = m_storage->m_dataType;

    m_bufferOffset = 0;
    m_bufferSize = GetDataSize();
}

Tensor::~Tensor()
{
}

Tensor MakeTensor(rad::ArrayRef<size_t> sizes, DataType dataType, const TensorOptions& options)
{
    Device* device = GetCurrentDevice();
    rad::Ref<TensorStorage> storage = device->CreateTensorStorage(sizes, dataType, options);
    return Tensor(storage, GetCurrentContext());
}

Tensor MakeTensorLike(const Tensor& ref)
{
    TensorOptions options = {};
    options.m_strides = ref.m_strides;
    return Tensor(ref.GetDevice()->CreateTensorStorage(ref.m_sizes, ref.m_dataType, options), ref.m_context);
}

Tensor MakeTensorLike(const Tensor* ref)
{
    return MakeTensorLike(*ref);
}

bool Tensor::IsFloatingPoint() const
{
    return IsFloatingPointType(m_dataType);
}

bool Tensor::IsInteger() const
{
    return IsIntegerType(m_dataType);
}

bool Tensor::IsSignedInteger() const
{
    return IsSignedIntegerType(m_dataType);
}

bool Tensor::IsUnsignedInteger() const
{
    return IsUnsignedIntegerType(m_dataType);
}

bool Tensor::IsBool() const
{
    return (m_dataType == DataType::Bool);
}

bool Tensor::IsComplex() const
{
    return IsComplexType(m_dataType);
}

size_t Tensor::GetElementCount() const
{
    return GetTensorElementCount(m_sizes);
}

std::vector<size_t> Tensor::GetMemoryOrder() const
{
    std::vector<size_t> order(m_strides.size());
    std::iota(order.begin(), order.end(), size_t(0));
    std::ranges::stable_sort(order,
        [&](size_t i, size_t j) {
            if (m_strides[i] < m_strides[j])
            {
                return true;
            }
            else if (m_strides[i] > m_strides[j])
            {
                return false;
            }
            else
            {
                return i > j;
            }
        }
    );
    return order;
}

size_t Tensor::GetDataSizeInElement() const
{
    return GetTensorDataSizeInElement(m_sizes, m_strides);
}

size_t Tensor::GetDataSize() const
{
    return GetDataSizeInElement() * GetElementSize(m_dataType);
}

bool Tensor::IsContiguous() const
{
    return (GetElementCount() == GetDataSizeInElement());
}

bool Tensor::HasSameLayout(const Tensor& other) const
{
    return HaveSameLayout(*this, other);
}

bool Tensor::HasSameLayout(const Tensor* other) const
{
    return HaveSameLayout(*this, *other);
}

void Tensor::Read(void* data, size_t offset, size_t dataSize)
{
    m_storage->Read(data, offset, dataSize);
}

void Tensor::Write(const void* data, size_t offset, size_t dataSize)
{
    m_storage->Write(data, offset, dataSize);
}

size_t Tensor::CoordToBufferIndex(rad::ArrayRef<size_t> coord) const
{
    assert(coord.size() == m_strides.size());
    return std::inner_product(coord.begin(), coord.end(), m_strides.begin(), size_t(0));
}

size_t Tensor::CoordToBufferOffset(rad::ArrayRef<size_t> coord) const
{
    return CoordToBufferIndex(coord) * GetElementSize(m_dataType);
}

bool Tensor::IsNCHW() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNHWC() const
{
    if ((m_sizes.size() == 4) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNCDHW() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[1]) &&
        (m_strides[1] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] == 1))
    {
        return true;
    }
    return false;
}

bool Tensor::IsNDHWC() const
{
    if ((m_sizes.size() == 5) &&
        (m_strides[0] > m_strides[2]) &&
        (m_strides[2] > m_strides[3]) &&
        (m_strides[3] > m_strides[4]) &&
        (m_strides[4] > m_strides[1]) &&
        (m_strides[1] == 1))
    {
        return true;
    }
    return false;
}

std::string Tensor::ToString(rad::ArrayRef<size_t> offsets, rad::ArrayRef<size_t> sizes, TextFormat format)
{
    if (m_sizes.empty())
    {
        return {};
    }
    std::string ss;
    ss.reserve(1024 * 1024);
    ss += std::format("#Sizes=[{}]; Strides=[{}]; DataType={};\n",
        rad::ToString(m_sizes, ", "), rad::ToString(m_strides, ", "), GetDataTypeName(m_dataType));
    std::vector<uint8_t> dataBuffer;
    dataBuffer.resize(GetDataSize());
    m_storage->Read(dataBuffer.data(), 0, dataBuffer.size());
    const uint8_t* data = dataBuffer.data();
    TensorIterator iter(this, offsets, sizes);
    size_t dimCount = m_sizes.size();

    rad::StringTable table;
    if (dimCount == 1)
    {
        size_t offsetW = iter.m_offsets[dimCount - 1];
        size_t sizeW = iter.m_sizes[dimCount - 1];
        table.Reserve(1, iter.m_sizes[0]);
        table.AddRow();
        for (size_t w = offsetW; w < offsetW + sizeW; ++w)
        {
            iter.m_coord[0] = w;
            table.AddCol(FormatDec(data + w * GetElementSize(m_dataType), m_dataType));
        }
        if (offsetW + sizeW < m_sizes[0])
        {
            table.AddCol("...");
        }
        rad::TableFormatter formatter(table);
        formatter.SetColAlignment(rad::TableFormatter::ColAlignment::Right);
        formatter.NormalizeColWidths(0, sizeW - 1);
        ss += formatter.Format();
        ss += '\n';
    }
    else
    {
        do {
            iter.Reset2D();
            size_t offsetH = iter.m_offsets[dimCount - 2];
            size_t offsetW = iter.m_offsets[dimCount - 1];
            size_t sizeH = iter.m_sizes[dimCount - 2];
            size_t sizeW = iter.m_sizes[dimCount - 1];
            ss += std::format("#Offsets=[{}]\n", rad::ToString(iter.m_coord, ", "));
            table.Clear();
            table.Reserve(sizeH, sizeW);
            for (size_t h = offsetH; h < offsetH + sizeH; ++h)
            {
                iter.m_coord[dimCount - 2] = h;
                table.AddRow();
                for (size_t w = offsetW; w < offsetW + sizeW; ++w)
                {
                    iter.m_coord[dimCount - 1] = w;
                    size_t bufferIndex = iter.m_bufferIndex + h * m_strides[dimCount - 2] + w * m_strides[dimCount - 1];
                    if (format == TextFormat::Dec)
                    {
                        table.AddCol(FormatDec(data + bufferIndex * GetElementSize(m_dataType), m_dataType));
                    }
                    else // if (format == TextFormat::Hex)
                    {
                        table.AddCol(FormatHex(data + bufferIndex * GetElementSize(m_dataType), m_dataType));
                    }
                }
                if (offsetW + sizeW < m_sizes[dimCount - 1])
                {
                    table.AddCol("...");
                }
            }
            rad::TableFormatter formatter(table);
            formatter.SetColAlignment(rad::TableFormatter::ColAlignment::Right);
            formatter.NormalizeColWidths(0, sizeW - 1);
            ss += formatter.Format();
        } while (iter.Next2D());
    }
    return ss;
}

Tensor& Tensor::Fill(const Scalar& value)
{
    m_context->Fill(*this, value);
    return *this;
}

Tensor Tensor::Add(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Add(*this, other, output);
    return output;
}

Tensor& Tensor::AddInPlace(const Scalar& other)
{
    m_context->Add(*this, other, *this);
    return *this;
}

Tensor Tensor::Add(const Tensor& other, const Scalar& alpha) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Add(*this, other, alpha, output);
    return output;
}

Tensor& Tensor::AddInPlace(const Tensor& other, const Scalar& alpha)
{
    m_context->Add(*this, other, alpha, *this);
    return *this;
}

Tensor Tensor::Subtract(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Subtract(*this, other, output);
    return output;
}

Tensor& Tensor::SubtractInPlace(const Scalar& other)
{
    m_context->Subtract(*this, other, *this);
    return *this;
}

Tensor Tensor::Subtract(const Tensor& other, const Scalar& alpha) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Subtract(*this, other, alpha, output);
    return output;
}

Tensor& Tensor::SubtractInPlace(const Tensor& other, const Scalar& alpha)
{
    m_context->Subtract(*this, other, alpha, *this);
    return *this;
}

Tensor Tensor::Multiply(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Multiply(*this, other, output);
    return output;
}

Tensor& Tensor::MultiplyInPlace(const Scalar& other)
{
    m_context->Multiply(*this, other, *this);
    return *this;
}

Tensor Tensor::Multiply(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Multiply(*this, other, output);
    return output;
}

Tensor& Tensor::MultiplyInPlace(const Tensor& other)
{
    m_context->Multiply(*this, other, *this);
    return *this;
}

Tensor Tensor::Divide(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Divide(*this, other, output);
    return output;
}

Tensor& Tensor::DivideInPlace(const Scalar& other)
{
    m_context->Divide(*this, other, *this);
    return *this;
}

Tensor Tensor::Divide(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Divide(*this, other, output);
    return output;
}

Tensor& Tensor::DivideInPlace(const Tensor& other)
{
    m_context->Divide(*this, other, *this);
    return *this;
}

Tensor Tensor::Remainder(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Remainder(*this, other, output);
    return output;
}

Tensor& Tensor::Remainder_(const Scalar& other)
{
    m_context->Remainder(*this, other, *this);
    return *this;
}

Tensor Tensor::Remainder(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->Remainder(*this, other, output);
    return output;
}

Tensor& Tensor::Remainder_(const Tensor& other)
{
    m_context->Remainder(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseAnd(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseAnd(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseAnd_(const Scalar& other)
{
    m_context->BitwiseAnd(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseAnd(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseAnd(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseAnd_(const Tensor& other)
{
    m_context->BitwiseAnd(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseNot() const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseNot(*this, output);
    return output;
}

Tensor& Tensor::BitwiseNot_()
{
    m_context->BitwiseNot(*this, *this);
    return *this;
}

Tensor Tensor::BitwiseOr(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseOr(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseOr_(const Scalar& other)
{
    m_context->BitwiseOr(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseOr(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseOr(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseOr_(const Tensor& other)
{
    m_context->BitwiseOr(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseXor(const Scalar& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseXor(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseXor_(const Scalar& other)
{
    m_context->BitwiseXor(*this, other, *this);
    return *this;
}

Tensor Tensor::BitwiseXor(const Tensor& other) const
{
    Tensor output = MakeTensorLike(this);
    m_context->BitwiseXor(*this, other, output);
    return output;
}

Tensor& Tensor::BitwiseXor_(const Tensor& other)
{
    m_context->BitwiseXor(*this, other, *this);
    return *this;
}

} // namespace ML
