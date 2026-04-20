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

#include "Core/Types.h"
#include "Interfaces/IModel.h"
#include <vector>

#include "Core/Rules.h"
#include "Interfaces/IQuat.h"

namespace NR
{
	template<FloatingPoint T = float>
	class Trainee
	{
		static constexpr int AmpWindow = 16;

		std::shared_ptr<IModel<T> > TargetModel;
		std::unique_ptr<torch::optim::Adam> Optimizer;

		std::vector<std::deque<float> > PredXHistory;
		std::vector<std::deque<float> > IdealXHistory;

		Rules Evaluator;
		NRModelProfile RigDesc;
		std::unordered_map<std::string, NRRule> V_rules;


	public:
		/**
		 * Calculates the next rotation for a given trainee in a training schedule.
		 *
		 * @param TargetModel The current trainee's identifier.
		 * @param Rig The list of all trainees in the schedule.
		 * @param LearningRate The step or interval to determine the next trainee.
		 * @param QCustom Optional custom quaternion converter for specific training scenarios.
		 */
		torch::Tensor IdealTargets;
		torch::Tensor Predicated;
		torch::Tensor PredHistory;
		torch::Tensor PredHistory2;
		torch::Tensor SmoothedOutput;
		IQuat* QuatConverter = nullptr;


		std::vector<torch::Tensor> PredictionCandidates;
		static constexpr size_t MaxCandidates = 10;

		torch::Tensor ChooseBestPrediction(
			const std::vector<torch::Tensor>& candidates,
			const torch::Tensor& target,
			const torch::Tensor& prevPred);

		/**
		 * Retrieves the next trainee in the training sequence based on the current state.
		 *
		 * @param TargetModel
		 * @param QCustom
		 * @param Rig
		 * @param Ev
		 * @param LearningRate
		 */
		Trainee(std::shared_ptr<IModel<T> > TargetModel, IQuat* QCustom, NRModelProfile Rig, Rules& Ev, double LearningRate = 1e-3);

		/**
		 * Executes the next step in the training process by determining the current stage.
		 *
		 * @param InputFloats The dataset being used for the training process.
		 * @return A flag indicating whether the training step was successfully completed.
		 */
		float TrainStep(const std::vector<float>& InputFloats);


		/**
		 * @brief Calculates all losses based on the training weights configuration (TW.json).
		 * @param Pred Neural network prediction
		 * @param Target Ideal IK target/ground truth
		 * @param Input Original network input
		 * @param PrevPred Previous frame prediction (for temporal loss)
		 * @return Detailed IKLossResult containing individual loss components
		 */
		IKLossResult ComputeLoss(const torch::Tensor& Pred, const torch::Tensor& Target, const torch::Tensor& Input, const torch::Tensor& PrevPred);

		/**
		 * @brief Performs Forward Kinematics to validate the skeleton hierarchy and bone lengths.
		 * @param Pred Neural network prediction
		 * @param Target Original network input (used to extract IK targets)
		 * @return Tensor containing FK error (bone chain integrity)
		 */
		torch::Tensor ComputeFK(const torch::Tensor& Pred, const torch::Tensor& Target);

		// ... existing code ...
		void Reset();

		void SaveWeights(const std::string& Path);

		void LoadWeights(const std::string& Path);
	};
} // namespace NR
