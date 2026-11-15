# =========================================================================
# Кросс-сборка нативного раннера KAN-ASR для Luckfox Pico Plus (RV1103).
#
# Требуется официальный Luckfox SDK (arm-rockchip830-linux-uclibcgnueabihf).
# Пример:
#   cmake -B build \
#     -DCMAKE_C_COMPILER=arm-rockchip830-linux-uclibcgnueabihf-gcc \
#     -DCMAKE_CXX_COMPILER=arm-rockchip830-linux-uclibcgnueabihf-g++ \
#     -DRKNN_SDK_ROOT=/path/to/luckfox/rknn
#   cmake --build build
# =========================================================================
cmake_minimum_required(VERSION 3.10)
project(luckfox_kan_asr CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Каталог с rknn_api.h и библиотекой librknnmrt / librknnrt.
# Задаётся через -DRKNN_SDK_ROOT=... либо переменную окружения.
if(NOT RKNN_SDK_ROOT)
    set(RKNN_SDK_ROOT "$ENV{RKNN_SDK_ROOT}")
endif()

if(NOT RKNN_SDK_ROOT)
    message(FATAL_ERROR
        "RKNN_SDK_ROOT не задан. Укажите путь к Luckfox RKNN SDK: "
        "-DRKNN_SDK_ROOT=/path/to/luckfox/rknn")
endif()

add_executable(luckfox_kan_asr luckfox_asr_inference.cpp)

target_include_directories(luckfox_kan_asr PRIVATE
    "${RKNN_SDK_ROOT}/include")

# В разных версиях SDK библиотека называется librknnmrt или librknnrt.
find_library(RKNN_LIB
    NAMES rknnmrt rknnrt
    PATHS "${RKNN_SDK_ROOT}/lib"
    NO_DEFAULT_PATH)

if(NOT RKNN_LIB)
    message(FATAL_ERROR
        "Не найдена библиотека rknnmrt/rknnrt в ${RKNN_SDK_ROOT}/lib")
endif()

target_link_libraries(luckfox_kan_asr PRIVATE "${RKNN_LIB}")

# Статическая линковка libstdc++/libgcc для uclibc-окружения Luckfox.
target_link_options(luckfox_kan_asr PRIVATE -static-libstdc++ -static-libgcc)
