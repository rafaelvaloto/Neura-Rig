// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRSolver/NRSolver.h"

namespace NR
{
	std::vector<float> NRSolver::Solve(const std::vector<float>& Inputs)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize() + RigDesc.GetRequiredTargetsSize();
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
