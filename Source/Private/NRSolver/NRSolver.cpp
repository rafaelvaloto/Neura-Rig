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
#include "NRSolver/NRSolver.h"

namespace NR
{
	std::vector<float> NRSolver::Solve(const std::vector<float>& Inputs)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t batchSize = static_cast<int32_t>(Inputs.size()) / InCount;

		torch::Tensor InputTensor = torch::from_blob((void*)Inputs.data(), {batchSize, InCount}, torch::kFloat32);

		torch::NoGradGuard NoGrad;
		torch::Tensor OutputTensor = NeuralNetwork->Forward(InputTensor).to(torch::kCPU);

		int32_t OutCount = RigDesc.GetRequiredOutputSize();
		std::vector<float> Results(OutCount * batchSize);
		std::memcpy(Results.data(), OutputTensor.data_ptr<float>(), OutputTensor.nbytes());
		return Results;
	}
} // namespace NR
