find_package(Protobuf REQUIRED)

set(NAME migraphx_test_runner)
set(TEST_DIR ${TEST_ONNX_DIR})
file(GLOB TEST_SRC "*.cpp")
add_executable(${NAME} ${TEST_SRC})
rocm_clang_tidy_check(${NAME})
target_link_libraries(${NAME} migraphx_c onnx-proto stdc++fs)
target_include_directories(${NAME} PUBLIC ../include include)
add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}> WORKING_DIRECTORY ${TEST_DIR}) 

