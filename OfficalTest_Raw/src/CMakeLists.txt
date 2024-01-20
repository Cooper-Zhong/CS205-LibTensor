add_library(${PROJECT_NAME} SHARED "")

file(GLOB_RECURSE SOURCES *.cpp *.cxx *.cc *.c)
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
