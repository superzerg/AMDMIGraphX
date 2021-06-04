cmake_minimum_required(VERSION 3.5)
project (migraphx_test_runner)

add_definitions(-D__HIP_PLATFORM_HCC__)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
#set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/MIGraph)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_debug)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)

set (DEPS_FOLDER /home/scxiao/Workplace/projects/MIGraph/deps_json)
set (CMAKE_CXX_STANDARD 17)

set (CMAKE_PREFIX_PATH /home/scxiao/Workplace/projects/MIGraph/deps_json)
find_package(Protobuf REQUIRED)

include_directories(${MIGRAPHX_FOLDER}/src/api/include
                    /opt/rocm-4.1.0/include
                    ${DEPS_FOLDER}/include)
link_directories(${MIGRAPHX_BUILD}/lib
                 ${DEPS_FOLDER}/lib)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})

file(GLOB src_files "*.cpp")
set (bin_name migraphx_test_runner)
message("source file: " ${src_name} " ---> bin: " ${bin_name})
add_executable(${bin_name} ${src_files})
target_link_libraries(${bin_name} migraphx_c onnx-proto stdc++fs)
