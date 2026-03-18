// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include <filesystem>
#include <iostream>
#include <string>

#include "NRCore/NRRules.h"

int main() {
    std::cout << "--- NeuraRig Logic Rule Integration Test ---" << std::endl;

    NR::NRRules evaluator;

    evaluator.Vars["velocity"] = 300.0f;
    evaluator.Vars["gain"] = 1.2f;
    evaluator.Vars["bone_l1"] = 45.0f;
    evaluator.Vars["bone_l2"] = 45.0f;
    evaluator.Vars["limit"] = 120.0f;
    evaluator.Vars["delta"] = 0.0f;
    evaluator.Vars["frequency"] = 0.5f;
    evaluator.Vars["offset"] = 0.0f;
    evaluator.Vars["cycle"] = 0.0f;
    evaluator.Vars["height"] = 25.0f;
    evaluator.Vars["progress"] = 0.0f;

    for (auto& [name, val] : evaluator.Vars) {
        evaluator.parser.DefineVar(name, &val);
    }

	std::cout << "DEBUG: delta parser: " << evaluator.Eval("delta") << std::endl;
    std::string stride_logic   = "min(velocity * gain, min((bone_l1 + bone_l2) * 1.2, limit))";
    std::string cycle_logic    = "fmod((delta * frequency) + offset, 1.0)";
    std::string progress_logic = "(cycle - 0.45) / 0.55";
    std::string z_logic        = "height * sin(_pi * progress)";

    std::cout << "Time | Cycle | Progress | Z_Offset" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    for (float t = 0; t <= 2.0f; t += 0.1f) {
        evaluator.Vars["delta"] = t;

        float stride = evaluator.Eval(stride_logic);
        float cycle = evaluator.Eval(cycle_logic);

        evaluator.Vars["cycle"] = cycle;

        float offset_z = 0.0f;
        float progress = 0.0f;

        if (cycle >= 0.45f) {
            progress = evaluator.Eval(progress_logic);
            evaluator.Vars["progress"] = progress;
            offset_z = evaluator.Eval(z_logic);
        }

        printf("%.1fs | %.2f  | %.2f     | %.2f\n", t, cycle, progress, offset_z);
    }

    std::cout << "--- Test Finished ---" << std::endl;
    return 0;
}
