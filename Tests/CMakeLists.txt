# 1. Find source files
set(TRAINEE_SOURCES "Integration/Trainee.cpp")
set(SERVER_SOURCES "Integration/Server.cpp")
set(RULES_SOURCES "Integration/Rules.cpp")

# 2. Create executables
add_executable(NRTestRules ${RULES_SOURCES})
add_executable(NRTestTrain ${TRAINEE_SOURCES})
add_executable(NRTestServer ${SERVER_SOURCES})

# 3. Configure compilation options
foreach(TARGET_NAME NRTestServer NRTestTrain NRTestRules)
    if (MSVC)
        # Opções gerais
        target_compile_options(${TARGET_NAME} PRIVATE /W4 /permissive-)

        # Desabilitar warnings chatos do LibTorch e do muparser
        target_compile_options(${TARGET_NAME} PRIVATE /wd4244 /wd4267 /wd4996 /wd4251 /wd4244)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -pedantic)
    endif ()

    # 4. Linkar bibliotecas
    target_link_libraries(${TARGET_NAME} PRIVATE NeuralRig ${TORCH_LIBRARIES})

    # 5. Copiar DLLs e Datasets no Windows (Pós-build)
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS}
                $<TARGET_FILE_DIR:${TARGET_NAME}>
                COMMENT "Copiando DLLs do Torch para ${TARGET_NAME}..."
                COMMAND_EXPAND_LISTS
        )

        add_custom_command(TARGET NRTestRules POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:muparser>" # Pega o caminho da DLL gerada pelo target muparser
                "$<TARGET_FILE_DIR:NRTestRules>" # Pasta onde o NRTestRules.exe está
        )

        # Copiar pasta Datasets para o diretório do executável
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/Datasets"
                "$<TARGET_FILE_DIR:${TARGET_NAME}>/Datasets"
                COMMENT "Copiando Datasets para o diretorio do ${TARGET_NAME}..."
        )
    endif()
endforeach()