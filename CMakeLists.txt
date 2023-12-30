cmake_minimum_required(VERSION 3.10)

project(Tensor)

set(CMAKE_CXX_STANDARD 17)

# 包含头文件目录
include_directories(Include)
message("Start processing...")

# 查找并包含所有源文件
file(GLOB SRC_FILES "Src/*.cpp")
file(GLOB TEST_FILES "Test/*.cpp")

# 添加可执行文件，每个测试文件都对应一个可执行文件
foreach(TEST_FILE ${TEST_FILES})
    # 输出与当前测试文件相关的源文件
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES})
endforeach()

