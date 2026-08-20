cmake_minimum_required(VERSION 3.10)
project(cube)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -O3 -march=native")

add_executable(cube cube.c)

target_link_libraries(cube 
    raylib 
    glfw 
    vulkan 
    m 
    pthread 
    dl
)
