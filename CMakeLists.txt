cmake_minimum_required(VERSION 3.10)

project(Tensor)

set(CMAKE_CXX_STANDARD 17)

# 包含头文件目录
include_directories(Include)
message("Start processing...")

# 查找并包含所有源文件
file(GLOB SRC_FILES "Src/*.cpp")
file(GLOB TEST_FILES "Test/*.cpp")
file(GLOB BENCHMARK_FILES "Benchmark/*.cpp")

# 添加可执行文件，每个测试文件都对应一个可执行文件
foreach(TEST_FILE ${TEST_FILES})
    # 输出与当前测试文件相关的源文件
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES})
endforeach()

find_package(benchmark REQUIRED)

# Benchmark 文件
foreach(BENCHMARK_FILE ${BENCHMARK_FILES})
    # 输出与当前测试文件相关的源文件
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE} ${SRC_FILES})
    target_link_libraries(${BENCHMARK_NAME} benchmark::benchmark)
    target_compile_options(${BENCHMARK_NAME} PRIVATE -O3 -lpthread)
endforeach()