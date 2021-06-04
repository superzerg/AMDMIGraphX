#ifndef __PARSE_TENSOR_HPP__
#define __PARSE_TENSOR_HPP__

#include <onnx.pb.h>
#include <string>
#include <migraphx/migraphx.hpp>

migraphx::argument parse_tensor(const onnx::TensorProto& t, std::vector<std::string>& input_data);

migraphx::argument parse_pb_file(const std::string& file_name,
                                 std::vector<std::string>& input_data);

std::vector<char> read_pb_file(const std::string& file_name);

#endif
