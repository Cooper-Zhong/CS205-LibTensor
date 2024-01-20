cmake_minimum_required(VERSION 3.10)

project(Tensor)

set(CMAKE_CXX_STANDARD 17)

# 查找CUDA并设置相关变量
find_package(CUDA)
file(GLOB CUDA_FILES "Src/*.cu")  # 新增 CUDA 文件的查找

# 配置 CUDA 相关设置
if(CUDA_FOUND)
    set(CMAKE_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

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
    if(CUDA_FOUND)
        add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES} ${CUDA_FILES})  # 添加 CUDA 文件
        target_link_libraries(${TEST_NAME} ${CUDA_LIBRARIES})
    else()
        add_executable(${TEST_NAME} ${TEST_FILE} ${SRC_FILES})  # 添加 CUDA 文件
    endif()
endforeach()



# Benchmark and CUDA Test 文件
if(CUDA_FOUND)
    file(GLOB CUDATEST_FILES "Test/*.cu")
    foreach(CUDATEST_FILE ${CUDATEST_FILES})
        # 输出与当前测试文件相关的源文件
        get_filename_component(CUDATEST_NAME ${CUDATEST_FILE} NAME_WE)
        add_executable(${CUDATEST_NAME} ${CUDATEST_FILE} ${SRC_FILES} ${CUDA_FILES})  # 添加 CUDA 文件
        target_link_libraries(${CUDATEST_NAME} benchmark::benchmark OpenMP::OpenMP_CXX)
        target_compile_options(${CUDATEST_NAME} PRIVATE -O3 -lpthread)
        target_link_libraries(${CUDATEST_NAME} ${CUDA_LIBRARIES})
    endforeach()

    file(GLOB BENCHMARK_FILES "Benchmark/*.cu")
    find_package(benchmark REQUIRED)
    find_package(OpenMP)
    foreach(BENCHMARK_FILE ${BENCHMARK_FILES})
        # 输出与当前测试文件相关的源文件
        get_filename_component(BENCHMARK_NAME ${BENCHMARK_FILE} NAME_WE)
        add_executable(${BENCHMARK_NAME} ${BENCHMARK_FILE} ${SRC_FILES} ${CUDA_FILES})  # 添加 CUDA 文件
        target_link_libraries(${BENCHMARK_NAME} benchmark::benchmark OpenMP::OpenMP_CXX)
        target_compile_options(${BENCHMARK_NAME} PRIVATE -lpthread)
        target_link_libraries(${BENCHMARK_NAME} ${CUDA_LIBRARIES})
    endforeach()
endif()
