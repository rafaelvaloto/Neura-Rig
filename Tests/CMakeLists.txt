# 1. Find source files
set(TRAINEE_SOURCES "Integration/Trainee.cpp")
set(SERVER_SOURCES "Integration/Server.cpp")

# 2. Create executables
add_executable(SimpleTrainer ${TRAINEE_SOURCES})
add_executable(NRServer ${SERVER_SOURCES})

# 3. Configure compilation options
foreach(TARGET_NAME SimpleTrainer NRServer)
    if (MSVC)
        # Opções gerais
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /permissive-)

        # Desabilitar warnings chatos do LibTorch
        target_compile_options(${TARGET_NAME} PRIVATE /wd4244 /wd4267 /wd4996)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
    endif ()

    # 4. Linkar bibliotecas
    target_link_libraries(${TARGET_NAME} PRIVATE NeuralRig ${TORCH_LIBRARIES})

    # 5. Copiar DLLs no Windows (Pós-build)
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "Copiando DLLs do Torch para ${TARGET_NAME}..."
                COMMAND_EXPAND_LISTS
        )
    endif()
endforeach()