find_package(Protobuf REQUIRED)

set(NAME migraphx_test_runner)
set(TEST_DIR ${TEST_ONNX_DIR})
file(GLOB TEST_SRC "*.cpp")
add_executable(${NAME} ${TEST_SRC})
rocm_clang_tidy_check(${NAME})
target_link_libraries(${NAME} migraphx_c onnx-proto stdc++fs)
target_include_directories(${NAME} PUBLIC ../include include)
add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> WORKING_DIRECTORY ${TEST_DIR}) 

#include_directories(${MIGRAPHX_FOLDER}/src/api/include
#                    /opt/rocm-4.1.0/include
#                    ${DEPS_FOLDER}/include)
#link_directories(${MIGRAPHX_BUILD}/lib
#                 ${DEPS_FOLDER}/lib)
#
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
#add_library(onnx-proto STATIC ${PROTO_SRCS})
#target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
#target_compile_options(onnx-proto PRIVATE -w)
#target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
#
#file(GLOB src_files "*.cpp")
#set (bin_name migraphx_test_runner)
#message("source file: " ${src_name} " ---> bin: " ${bin_name})
#add_executable(${bin_name} ${src_files})
#target_link_libraries(${bin_name} migraphx_c onnx-proto stdc++fs)
