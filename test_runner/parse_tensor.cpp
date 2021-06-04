#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <hip/hip_fp16.h>
#include <cassert>
#include "parse_tensor.hpp"

using node_map = std::unordered_map<std::string, onnx::NodeProto>;

template<class T>
static void print(std::ostream& os, const std::vector<T>& dims)
{
    os << "{";
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        if (i != 0) os << ", ";
        os << dims[i];
    }
    os << "}";
}

template<class T>
static std::ostream& operator << (std::ostream& os, const std::vector<T>& dims)
{
    print(os, dims);
    return os;
}

template<class T>
migraphx::argument create_argument(migraphx_shape_datatype_t type, 
                                   const std::vector<std::size_t>& dims, 
                                   const std::vector<T>& data)
{
    migraphx::shape s(type, dims);
    return {s, (void*)data.data()};
}

migraphx::argument create_argument(migraphx_shape_datatype_t type, 
                                   const std::vector<std::size_t>& dims, 
                                   const char* data)
{
    migraphx::shape s(type, dims);
    return {s, (void*)data};
}

std::vector<char> read_pb_file(const std::string& filename)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = is.tellg();
    if(size < 1)
    {
        std::cout << "Invalid size for: " << filename << std::endl;
        std::abort();
    }
    is.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if(!is.read(buffer.data(), size))
    {
        std::cout << "Error reading file: " << filename << std::endl;
        std::abort();
    }
    return buffer;
}


migraphx_shape_datatype_t get_type(int dtype)
{
    switch(dtype)
    {
    case 1: return migraphx_shape_float_type;
    case 2: return migraphx_shape_uint8_type;
    case 3: return migraphx_shape_int8_type;
    case 4: return migraphx_shape_uint16_type;
    case 5: return migraphx_shape_int16_type;
    case 6: return migraphx_shape_int32_type;
    case 7: return migraphx_shape_int64_type;
    case 9: return migraphx_shape_bool_type;
    case 10: return migraphx_shape_half_type;
    case 11: return migraphx_shape_double_type;
    case 12: return migraphx_shape_uint32_type;
    case 13: return migraphx_shape_uint64_type;
    default: { 
		std::cout << "Prototensor data type " << std::to_string(dtype) << " not supported" << std::endl;
    }
    }
}

migraphx::argument parse_tensor(const onnx::TensorProto& t, std::vector<std::string>& input_data)
{
    std::vector<std::size_t> dims(t.dims().begin(), t.dims().end());
    if(not t.external_data().empty())
    {
        const std::string& data_file = t.external_data().at(0).value();
        std::string path = ".";
        auto raw_buffer              = read_pb_file(path + "/" + data_file);
        std::string s(raw_buffer.begin(), raw_buffer.end());
        input_data.push_back(s);
        auto type = get_type(t.data_type());
        return create_argument(type, dims, input_data.back().data());
    }
    if(t.has_raw_data())
    {
        const std::string& s = t.raw_data();
        input_data.push_back(s);
        auto type            = get_type(t.data_type());
        return create_argument(type, dims, input_data.back().data());
    }

    switch(t.data_type())
    {
    case onnx::TensorProto::BOOL:
    {
        std::vector<char> data(t.int32_data().begin(), t.int32_data().end());
		return create_argument(migraphx_shape_bool_type, dims, data);
    }
    case onnx::TensorProto::INT8:
    {
        std::vector<char> data(t.int32_data().begin(), t.int32_data().end());
        return create_argument(migraphx_shape_int8_type, dims, data);
    }
    case onnx::TensorProto::UINT8: 
    {
        std::vector<unsigned char> data(t.int32_data().begin(), t.int32_data().end());
        return create_argument(migraphx_shape_uint8_type, dims, data);
    }
    case onnx::TensorProto::INT16: 
    {
        std::vector<int16_t> data(t.int32_data().begin(), t.int32_data().end());
        return create_argument(migraphx_shape_int16_type, dims, data);
    }
    case onnx::TensorProto::UINT16: 
    {
        std::vector<uint16_t> data(t.int32_data().begin(), t.int32_data().end());
        return create_argument(migraphx_shape_uint16_type, dims, data);
    }
    case onnx::TensorProto::INT32: 
    {
        std::vector<int32_t> data(t.int32_data().begin(), t.int32_data().end());
        return create_argument(migraphx_shape_int32_type, dims, data);
    }
    case onnx::TensorProto::UINT32:
    {
        std::vector<uint32_t> data(t.int64_data().begin(), t.int64_data().end());
        return create_argument(migraphx_shape_uint32_type, dims, data);
    }
    case onnx::TensorProto::INT64: 
    {
        std::vector<int64_t> data(t.int64_data().begin(), t.int64_data().end());
        return create_argument(migraphx_shape_int64_type, dims, data);
    }
    case onnx::TensorProto::UINT64:
    {
        std::vector<uint64_t> data(t.uint64_data().begin(), t.uint64_data().end());
        return create_argument(migraphx_shape_uint64_type, dims, data);
    }
    case onnx::TensorProto::FLOAT16:
    {
        std::vector<uint16_t> data_uint16(t.int32_data().begin(), t.int32_data().end());
        std::vector<half> data_half;
        std::transform(data_uint16.begin(),
                       data_uint16.end(),
                       std::back_inserter(data_half),
                       [](uint16_t raw_val) { return *reinterpret_cast<half*>(&raw_val); });
        return create_argument(migraphx_shape_half_type, dims, data_half);
    }
    case onnx::TensorProto::DOUBLE:
    {
        std::vector<double> data(t.double_data().begin(), t.double_data().end());
        return create_argument(migraphx_shape_double_type, dims, data);
    }
    case onnx::TensorProto::FLOAT: 
    {
        std::vector<float> data(t.float_data().begin(), t.float_data().end());
        return create_argument(migraphx_shape_float_type, dims, data);
    }
    case onnx::TensorProto::UNDEFINED:
    case onnx::TensorProto::STRING:
    case onnx::TensorProto::COMPLEX64:
    case onnx::TensorProto::COMPLEX128: throw std::runtime_error("");
    }

    std::abort();
}

migraphx::argument parse_pb_file(const std::string& file_name, std::vector<std::string>& input_data)
{
    std::fstream input(file_name.c_str(), std::ios::in | std::ios::binary);
	if (!input.is_open())
    {
        std::cout << "Tensor File " << file_name << " open error!" << std::endl;
        std::abort();
    }

    onnx::TensorProto tensor;
    if (not tensor.ParseFromIstream(&input))
    {
        std::cout << "Parse tensor from file " << file_name << " error!" << std::endl;
        std::abort();
    }

    return parse_tensor(tensor, input_data);
}
