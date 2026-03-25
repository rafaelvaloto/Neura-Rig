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
#pragma once

#include "NRCore/NRTypes.h"
#include "NRInterfaces/INRModel.h"
#include <vector>

#include "NRCore/NRRules.h"

namespace NR
{
	template<FloatingPoint T = float>
	class NRTrainee
	{
		static constexpr int AmpWindow = 16;

		std::shared_ptr<INRModel<T> > TargetModel;
		std::unique_ptr<torch::optim::Adam> Optimizer;

		std::vector<std::deque<float> > PredXHistory;
		std::vector<std::deque<float> > IdealXHistory;

		NRRules Evaluator;
		NRModelProfile RigDesc;
		std::unordered_map<std::string, NRRule> V_rules;

	public:
		/**
		 * Calculates the next rotation for a given trainee in a training schedule.
		 *
		 * @param TargetModel The current trainee's identifier.
		 * @param Rig The list of all trainees in the schedule.
		 * @param LearningRate The step or interval to determine the next trainee.
		 */
		torch::Tensor IdealTargets;
		torch::Tensor Predicated;

		/**
		 * Retrieves the next trainee in the training sequence based on the current state.
		 *
		 * @param TargetModel
		 * @param Rig
		 * @param Ev
		 * @param LearningRate
		 */
		NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, NRModelProfile  Rig, NRRules& Ev, double LearningRate = 1e-3);

		/**
		 * Executes the next step in the training process by determining the current stage.
		 *
		 * @param InputFloats The dataset being used for the training process.
		 * @return A flag indicating whether the training step was successfully completed.
		 */
		float TrainStep(const std::vector<float>& InputFloats);

		/**
		 * Computes the Inverse Kinematics (IK) loss for a given model configuration.
		 *
		 * @param InputTensor The target pose to compare against.
		 * @param Pred The target pose to compare against.
		 * @return IKLossResult
		 */
		IKLossResult ComputeRLReward(const torch::Tensor& InputTensor, const torch::Tensor& Pred);

		/**
		 * @brief Resets all internal state to a clean slate for a fresh training session.
		 * Clears optimizer momentum, evaluator state, prediction history, and re-initializes model weights.
		 */
		void Reset();

		void SaveWeights(const std::string& Path);

		void LoadWeights(const std::string& Path);
	};
} // namespace NR
