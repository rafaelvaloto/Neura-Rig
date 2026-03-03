#include <filesystem>
#include <iostream>
#include <string>

#include "NRCore/NRRules.h"

int main() {
    std::cout << "--- NeuraRig Logic Integration Test ---" << std::endl;

    NR::NRRules evaluator;

    // 1. Inicialização das variáveis
    evaluator.Vars["velocity"] = 300.0f;
    evaluator.Vars["gain"] = 1.2f;
    evaluator.Vars["bone_l1"] = 45.0f;
    evaluator.Vars["bone_l2"] = 45.0f;
    evaluator.Vars["limit"] = 120.0f;
    evaluator.Vars["delta"] = 0.0f; // Vamos mudar isso no loop
    evaluator.Vars["frequency"] = 0.5f;
    evaluator.Vars["offset"] = 0.0f;
    evaluator.Vars["cycle"] = 0.0f;
    evaluator.Vars["height"] = 25.0f;
    evaluator.Vars["progress"] = 0.0f;

    // 2. Setup do Parser (Vincular endereços)
    for (auto& [name, val] : evaluator.Vars) {
        evaluator.parser.DefineVar(name, reinterpret_cast<mu::value_type*>(&val));
    }

    // 3. Definição das Strings de Lógica (Exatamente como no seu JSON)
    std::string stride_logic   = "min(velocity * gain, min((bone_l1 + bone_l2) * 1.2, limit))";
    std::string cycle_logic    = "fmod((delta * frequency) + offset, 1.0)";
    std::string progress_logic = "(cycle - 0.45) / 0.55";
    std::string z_logic        = "height * sin(_pi * progress)";

    // 4. Loop de Simulação (Simulando 2 segundos de caminhada)
    std::cout << "Time | Cycle | Progress | Z_Offset" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    for (float t = 0; t <= 2.0f; t += 0.1f) {
        evaluator.Vars["delta"] = t; // Atualiza o tempo

        // Executa a cadeia de lógica
        float stride = evaluator.Eval(stride_logic);
        float cycle = evaluator.Eval(cycle_logic);

        evaluator.Vars["cycle"] = cycle; // Injeta para o próximo cálculo

        float offset_z = 0.0f;
        float progress = 0.0f;

        if (cycle >= 0.45f) { // Fase de Swing (Pé no ar)
            progress = evaluator.Eval(progress_logic);
            evaluator.Vars["progress"] = progress;
            offset_z = evaluator.Eval(z_logic);
        }

        // Print formatado para ver o movimento
        printf("%.1fs | %.2f  | %.2f     | %.2f\n", t, cycle, progress, offset_z);
    }

    std::cout << "--- Test Finished ---" << std::endl;
    return 0;
}
