cmake_minimum_required(VERSION 3.10)

project(Tensor)

set(CMAKE_CXX_STANDARD 17)

# aux_source_directory(. DIR_SRCS)

include_directories(include)


file(GLOB SOURCES "src/*.cpp")


add_executable(Tensor ${SOURCES})

