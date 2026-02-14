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
		NRRigDescription RigDesc;

		/**
		 * Calculates the next rotation for a given trainee in a training schedule.
		 *
		 * @param TargetModel The current trainee's identifier.
		 * @param Rig The list of all trainees in the schedule.
		 * @param LearningRate The step or interval to determine the next trainee.
		 */
	public:
		NRTrainee(std::shared_ptr<INRModel<T>> TargetModel, NRRigDescription Rig, double LearningRate = 1e-3);

		/**
		 * Determines the next step for a trainee in the training process.
		 *
		 * @param InputVectors The current step number in the training sequence.
		 * @param TargetVectors The total number of steps in the training program.
		 * @return The next step number for the trainee, or a completion status if all steps are finished.
		 */
		float TrainStep(const std::vector<NRVector3D>& InputVectors, const std::vector<NRVector3D>& TargetVectors);

		void SaveWeights(const std::string& Path);
		void LoadWeights(const std::string& Path);
	};
} // namespace NR
