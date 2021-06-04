#ifndef MIGRAPHX_GUARD_TEST_RUNNER_PARSE_TENSOR_HPP
#define MIGRAPHX_GUARD_TEST_RUNNER_PARSE_TENSOR_HPP

#include <onnx.pb.h>
#include <string>
#include <migraphx/migraphx.hpp>

namespace onnx = onnx_for_migraphx;

migraphx::argument parse_tensor(const onnx::TensorProto& t, std::vector<std::string>& input_data);

migraphx::argument parse_pb_file(const std::string& file_name,
                                 std::vector<std::string>& input_data);

std::vector<char> read_pb_file(const std::string& file_name);

#endif
