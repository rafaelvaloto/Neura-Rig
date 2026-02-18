// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRSolver/NRSolver.h"

namespace NR
{
	std::vector<NR::NRVector3D> NRSolver::Solve(const std::vector<NR::NRVector3D>& Inputs)
	{
		// 1. Criar o Tensor a partir dos vetores de entrada (Pés)
		// {1, TotalFloats} -> Ex: 1 linha por 6 colunas (2 pés)
		int64_t inputSize = Inputs.size() * 3;
		torch::Tensor InputTensor = torch::from_blob((void*)Inputs.data(), {1, inputSize}, torch::kFloat32).to(Device);

		torch::NoGradGuard NoGrad;
		torch::Tensor OutputTensor = NeuralNetwork->Forward(InputTensor).to(torch::kCPU);

		std::vector<NR::NRVector3D> Results(RigDesc.TargetCount);
		std::memcpy(Results.data(), OutputTensor.data_ptr<float>(), OutputTensor.nbytes());
		return Results;
	}
} // namespace NR
