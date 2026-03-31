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
		NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, NRModelProfile Rig, NRRules& Ev, double LearningRate = 1e-3);

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

		[[nodiscard]] FootValidationPair ValidateFeetFK(
			const torch::Tensor& pred,
			const std::vector<int64_t>& leftOffsets,
			const std::vector<int64_t>& rightOffsets,
			bool anglesAreDegrees = true)
		{
			FootValidationPair result;
			result.err_loss = torch::tensor(0.0f, pred.options());

			auto computeFoot = [&](const std::vector<int64_t>& rotationOffsets, const torch::Tensor& footIKOffset, bool isRightLeg, int thighBindingIdx, int calfBindingIdx) {
				if (rotationOffsets.size() < 2)
					return;

				auto readRotWithLimit = [&](int64_t offset, int bindingIdx, bool applyThighOffset = false) -> torch::Tensor {
					if (offset + 3 > pred.size(1))
						return torch::zeros({3}, pred.options());
					auto rTensor = pred.index({0, torch::indexing::Slice(offset, offset + 3)}).clone();

					if (anglesAreDegrees)
					{
						rTensor = rTensor * (3.14159265358979323846f / 180.0f);
					}

					// Aplica offset de -180° no pitch do thigh direito
					if (applyThighOffset && isRightLeg)
					{
						rTensor[0] = rTensor[0] - 3.14159265358979323846f;
					} else if (applyThighOffset && !isRightLeg)
					{
						rTensor[1] = -rTensor[1];
					}

					// Apply Rotation Limits if available
					if (bindingIdx >= 0 && bindingIdx < RigDesc.Bindings.size())
					{
						const auto& binding = RigDesc.Bindings[bindingIdx];
						if (!binding.Rules.empty())
						{
							const auto& limits = binding.Rules[0].Limits;
							auto toRad = [](float deg) { return deg * (3.14159265358979323846f / 180.0f); };

							auto minBound = torch::tensor({toRad(limits.MinX), toRad(limits.MinY), toRad(limits.MinZ)}, rTensor.options());
							auto maxBound = torch::tensor({toRad(limits.MaxX), toRad(limits.MaxY), toRad(limits.MaxZ)}, rTensor.options());

							auto diffMin = torch::clamp(minBound - rTensor, 0.0f);
							auto diffMax = torch::clamp(rTensor - maxBound, 0.0f);
							result.err_loss = result.err_loss + (diffMin.pow(2).sum() + diffMax.pow(2).sum()) * 10.0f;

							rTensor = torch::clamp(rTensor, minBound, maxBound);
						}
					}

					return rTensor;
				};

				auto p0_tensor = readRotWithLimit(rotationOffsets[1], thighBindingIdx, true); // Thigh
				auto p1_tensor = readRotWithLimit(rotationOffsets[0], calfBindingIdx, false); // Calf

				// Bone lengths
				float L1 = 0.457519f;
				float L2 = 0.417055f;

				auto pelvisOffset = pred.index({0, torch::indexing::Slice(48, 48 + 3)});
				auto pelvisRotation = pred.index({0, torch::indexing::Slice(48 + 3, 48 + 6)});

				auto pelvisBasePos = torch::tensor({0.0f, 0.0f, 95.0f}, pred.options());
				auto hipPos = pelvisBasePos + pelvisOffset;

				auto getRotX = [&](torch::Tensor a) {
					auto c = torch::cos(a);
					auto s = torch::sin(a);
					return torch::stack({
						torch::stack({torch::ones({}, a.options()), torch::zeros({}, a.options()), torch::zeros({}, a.options())}),
						torch::stack({torch::zeros({}, a.options()), c, -s}),
						torch::stack({torch::zeros({}, a.options()), s, c})
					});
				};

				auto getRotY = [&](torch::Tensor a) {
					auto c = torch::cos(a);
					auto s = torch::sin(a);
					return torch::stack({
						torch::stack({c, torch::zeros({}, a.options()), s}),
						torch::stack({torch::zeros({}, a.options()), torch::ones({}, a.options()), torch::zeros({}, a.options())}),
						torch::stack({-s, torch::zeros({}, a.options()), c})
					});
				};

				auto getRotZ = [&](torch::Tensor a) {
					auto c = torch::cos(a);
					auto s = torch::sin(a);
					return torch::stack({
						torch::stack({c, -s, torch::zeros({}, a.options())}),
						torch::stack({s, c, torch::zeros({}, a.options())}),
						torch::stack({torch::zeros({}, a.options()), torch::zeros({}, a.options()), torch::ones({}, a.options())})
					});
				};

				auto getEulerXYZ = [&](torch::Tensor rot) {
					return torch::mm(torch::mm(getRotX(rot[0]), getRotY(rot[1])), getRotZ(rot[2]));
				};

				auto pelvisBaseRot = torch::tensor({0.0f, -1.570796f, 0.0f}, pred.options());
				auto mPelvis = torch::mm(getEulerXYZ(pelvisBaseRot), getEulerXYZ(pelvisRotation));

				// Forward Kinematics (differentiable)
				auto mThigh = torch::mm(mPelvis, getEulerXYZ(p0_tensor)); // Multiplica pelvis * thigh
				auto bone1 = torch::tensor({-L1, 0.0f, 0.0f}, pred.options()).unsqueeze(1);
				auto kneePos = hipPos + torch::mm(mThigh, bone1).squeeze(1);

				auto mCalf = getEulerXYZ(p1_tensor);
				auto mGlobalCalf = torch::mm(mThigh, mCalf);
				auto bone2 = torch::tensor({-L2, 0.0f, 0.0f}, pred.options()).unsqueeze(1);
				auto footPos = kneePos + torch::mm(mGlobalCalf, bone2).squeeze(1);

				// Target = AI predicate foot target
				auto footTarget = hipPos + footIKOffset;

				std::cout << "Foot Pos (FK): " << footPos[0] << ", " << footPos[1] << ", " << footPos[2] << std::endl;
				std::cout << "Foot Target: " << footTarget[0] << ", " << footTarget[1] << ", " << footTarget[2] << std::endl;
				std::cout << "Error: " << torch::norm(footPos - footTarget).item<float>() << " meters" << std::endl;

				// Position Error: "That are rotation be is correct?"
				auto posErrTensor = torch::norm(footPos - footTarget);
				result.err_loss = result.err_loss + posErrTensor;

				// Knee Angle consistency
				float d0 = std::abs(L1 - L2);
				float d1 = L1 + L2;
				auto D_tensor = torch::abs(footPos[0] - hipPos[0]);
				auto D_clamped = torch::clamp(D_tensor, d0, d1);

				auto cosAngle_tensor = (L1 * L1 + L2 * L2 - torch::pow(D_clamped, 2)) / (2 * L1 * L2);
				auto kneeAngle_tensor = torch::acos(torch::clamp(cosAngle_tensor, -1.0f, 1.0f));

				float footThreshold = 0.05f;
				auto errorWeight = torch::where(posErrTensor > footThreshold, torch::tensor(1.0f, pred.options()), posErrTensor / footThreshold);

				auto p0_x_abs = torch::abs(p0_tensor[0]);
				auto p1_x_abs = torch::abs(p1_tensor[0]);
				auto kneeAngle_abs = torch::abs(kneeAngle_tensor);

				auto knee_consistency_loss = (torch::abs(kneeAngle_abs - p0_x_abs) + torch::abs(kneeAngle_abs - p1_x_abs)) * errorWeight;
				result.err_loss = result.err_loss + knee_consistency_loss * 0.1f;
			};

			auto P1_offset = pred.index({0, torch::indexing::Slice(0, 3)});
			auto P2_offset = pred.index({0, torch::indexing::Slice(6, 9)});

			auto findBindingIdx = [&](const std::string& name) -> int {
				for (int i = 0; i < (int)RigDesc.Bindings.size(); ++i)
				{
					if (RigDesc.Bindings[i].BoneName == name)
						return i;
				}
				return -1;
			};

			computeFoot(rightOffsets, P1_offset, true, findBindingIdx("thigh_r"), findBindingIdx("calf_r"));
			computeFoot(leftOffsets, P2_offset, false, findBindingIdx("thigh_l"), findBindingIdx("calf_l"));

			return result;
		}

		/**
		 * @brief Resets all internal state to a clean slate for a fresh training session.
		 * Clears optimizer momentum, evaluator state, prediction history, and re-initializes model weights.
		 */
		void Reset();

		void SaveWeights(const std::string& Path);

		void LoadWeights(const std::string& Path);
	};
} // namespace NR
