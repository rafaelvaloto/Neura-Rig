# 1. Encontrar os arquivos fonte
file(GLOB_RECURSE TEST_SOURCES "Integration/*.cpp")

# 2. PRIMEIRO: Criar o executável (Só uma vez!)
add_executable(SimpleTrainer ${TEST_SOURCES})

# 3. AGORA SIM: Configurar opções de compilação (Depois de criar o alvo)
if (MSVC)
    # Opções gerais
    target_compile_options(SimpleTrainer PRIVATE /W4 /permissive-)

    # Desabilitar warnings chatos do LibTorch
    target_compile_options(SimpleTrainer PRIVATE /wd4244 /wd4267 /wd4996)
else()
    target_compile_options(SimpleTrainer PRIVATE -Wall -Wextra -pedantic)
endif ()

# 4. Linkar bibliotecas
target_link_libraries(SimpleTrainer PRIVATE NeuralRig ${TORCH_LIBRARIES})

# 5. Copiar DLLs no Windows (Pós-build)
if(MSVC)
    # Copia DLLs do Torch para a pasta do executável automaticamente
    add_custom_command(TARGET SimpleTrainer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TORCH_INSTALL_PREFIX}/lib/torch_cpu.dll"
            "${TORCH_INSTALL_PREFIX}/lib/c10.dll"
            "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
            # Adicione outras DLLs se necessário (intel_omp, etc)
            $<TARGET_FILE_DIR:SimpleTrainer>
            COMMENT "Copiando DLLs do Torch para o SimpleTrainer..."
            COMMAND_EXPAND_LISTS
    )

    # Maneira alternativa (Copia tudo *.dll da pasta lib) - O seu jeito anterior
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET SimpleTrainer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:SimpleTrainer>
            COMMENT "Copiando todas as DLLs do Torch..."
            COMMAND_EXPAND_LISTS
    )
endif()