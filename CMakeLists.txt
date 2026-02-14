cmake_minimum_required(VERSION 3.20)

project(NeuralRig LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- LIBTORCH (3rdParty) ---
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/libtorch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "LibTorch: ${TORCH_INSTALL_PREFIX}")

add_subdirectory(Source)
add_subdirectory(Tests)
