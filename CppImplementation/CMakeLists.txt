cmake_minimum_required(VERSION 3.14)
project(ArbitrageBacktester CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(arbitrage_backtester
    src/main.cpp
    src/ArbitrageBacktester.cpp
)

target_include_directories(arbitrage_backtester PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(arbitrage_backtester PRIVATE -O3)
endif()
