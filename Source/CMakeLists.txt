file(GLOB_RECURSE LIB_SOURCES
        "Private/*.cpp"
        "Public/*.h"
)

add_library(NeuralRig STATIC ${LIB_SOURCES})

target_include_directories(NeuralRig
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/Public
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Private
)

target_link_libraries(NeuralRig PUBLIC ${TORCH_LIBRARIES})