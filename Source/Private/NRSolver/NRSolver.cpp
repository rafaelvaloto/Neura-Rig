// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRSolver/NRSolver.h"

namespace NR
{
	std::vector<NRVector3D> NRSolver::Solve(const std::vector<NRVector3D>& Inputs)
	{
		// 1. Criar o Tensor a partir dos vetores de entrada
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t batchSize = (static_cast<int32_t>(Inputs.size()) * 3) / InCount;

		torch::Tensor InputTensor = torch::from_blob((void*)Inputs.data(), {batchSize, InCount}, torch::kFloat32);

		torch::NoGradGuard NoGrad;
		torch::Tensor OutputTensor = NeuralNetwork->Forward(InputTensor).to(torch::kCPU);

		int32_t OutCount = RigDesc.GetRequiredOutputSize();
		std::vector<NRVector3D> Results(OutCount / 3);
		std::memcpy(Results.data(), OutputTensor.data_ptr<float>(), OutputTensor.nbytes());
		return Results;
	}
} // namespace NR
