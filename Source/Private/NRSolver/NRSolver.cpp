// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRSolver/NRSolver.h"

namespace NR
{
	void NRSolver::solve(float x, float y, float z)
	{
		torch::Tensor Input = torch::tensor({x, y, z});
		torch::Tensor Output = NeuralNetwork->Forward(Input);
		ApplyToRig(Output);
	}

	void NRSolver::ApplyToRig(torch::Tensor Angles)
	{
	}
} // namespace NR
