cmake_minimum_required(VERSION 3.5)
project (CAI)

set (CMAKE_CXX_STANDARD 14)
set (EXAMPLE mnist_inference)

list (APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package (migraphx)

message("source file: " ${EXAMPLE}.cpp " ---> bin: " ${EXAMPLE})
add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
target_link_libraries(${EXAMPLE} migraphx::c)

add_executable(mnist mnist2.cpp)
target_link_libraries(mnist migraphx::c)

