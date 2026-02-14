// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once

#include "NRCore/NRTypes.h"
#include "NRInterfaces/INRModel.h"
#include <vector>

namespace NR
{
	template<FloatingPoint T = float>
	class NRTrainee
	{
		std::shared_ptr<INRModel<T>> TargetModel;
		std::unique_ptr<torch::optim::Adam> Optimizer;

	public:
		NRTrainee(std::shared_ptr<INRModel<T>> TargetModel, double LearningRate = 1e-3);

		float TrainStep(const std::vector<NRVector3D>& InputVectors, const std::vector<NRVector3D>& TargetVectors);
	};
} // namespace NR
