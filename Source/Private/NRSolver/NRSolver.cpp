// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRSolver/NRSolver.h"

namespace NR
{
	std::vector<NRVector3D> NRSolver::Solve(const std::vector<NRVector3D>& Targets)
	{

		torch::Tensor InputTensor = torch::from_blob((void*)Targets.data(), {1, RigDesc.GetRequiredInputSize()}, torch::kFloat32).to(Device);

		torch::NoGradGuard NoGrad;
		torch::Tensor OutputTensor = NeuralNetwork->Forward(InputTensor).to(torch::kCPU);

		std::vector<NRVector3D> Results(RigDesc.BoneCount);
		std::memcpy(Results.data(), OutputTensor.data_ptr<float>(), OutputTensor.nbytes());
		return Results;
	}
} // namespace NR
