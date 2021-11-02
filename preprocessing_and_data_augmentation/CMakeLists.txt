cmake_minimum_required(VERSION 3.16)
project(vis_c)

find_package(PCL 1.11 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

find_package(fmt)

set(CMAKE_CXX_STANDARD 17)

add_executable(vis_c main.cpp)
target_link_libraries(vis_c ${PCL_LIBRARIES} ${Boost_LIBRARIES} fmt::fmt)
