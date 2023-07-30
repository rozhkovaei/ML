cmake_minimum_required(VERSION 3.10)

project(fashion_mnist CXX)

add_subdirectory(mnist)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    mnist/include/mnist/
)

add_executable(${PROJECT_NAME} main.cpp
    mnist/demo/helpers.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${HRDS}
    mnist/demo
)

target_link_libraries(${PROJECT_NAME}
    mnist
)