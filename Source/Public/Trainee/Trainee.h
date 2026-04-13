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
		 */
		torch::Tensor IdealTargets;
		torch::Tensor Predicated;
		torch::Tensor PredHistory;
		torch::Tensor PredHistory2;
		torch::Tensor SmoothedOutput;

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
		 * @param Rig
		 * @param Ev
		 * @param LearningRate
		 */
		Trainee(std::shared_ptr<IModel<T> > TargetModel, NRModelProfile Rig, Rules& Ev, double LearningRate = 1e-3);

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

		// ... existing code ...
		[[nodiscard]] FootValidationPair ValidateFeetFK(
			const torch::Tensor& pred,
			const torch::Tensor& target,
			const torch::Tensor& input
			)
		{
			FootValidationPair result;
			result.err_loss = torch::tensor(0.0f, pred.options());

			// Helper: quaternion [x,y,z,w] -> 3x3 rotation matrix
			auto quatToMat = [&](const torch::Tensor& q) -> torch::Tensor {
				auto qn = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
				float x = qn[0].item<float>(), y = qn[1].item<float>();
				float z = qn[2].item<float>(), w = qn[3].item<float>();
				return torch::tensor({
					{1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y},
					{2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x},
					{2*x*z - 2*w*y,       2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y}
				}, pred.options());
			};

			// Helper: get initial (bind-pose) quat for a binding index
			auto getBindQuat = [&](int idx) -> torch::Tensor {
				if (idx < 0 || idx >= (int)RigDesc.Bindings.size())
				{
					return torch::tensor({0.f, 0.f, 0.f, 1.f}, pred.options());
				}

				const auto& b = RigDesc.Bindings[idx];
				auto it = std::find_if(b.Rules.begin(), b.Rules.end(), [&](const auto& r){ return r.Name == b.RuleName; });
				if (it == b.Rules.end() || it->Phases.empty())
				{
					return torch::tensor({0.f, 0.f, 0.f, 1.f}, pred.options());
				}

				const auto& ph = it->Phases[0];
				float q1=0, q2=0, q3=0, qw=1;
				for (const auto& f : ph.Formulas) {
					try {
						// Try to parse as float, if it's a variable or formula, use Evaluator
						float val = 0.0f;
						try {
							val = std::stof(f.Expr);
						} catch (...) {
							val = (float)Evaluator.Eval(idx, f.Expr);
						}

						if (f.Name == "q1") q1 = val;
						else if (f.Name == "q2") q2 = val;
						else if (f.Name == "q3") q3 = val;
						else if (f.Name == "qw") qw = val;
					} catch (...)
					{
						std::cerr  << "Failed to parse formula: " << f.Name << " = " << f.Expr << std::endl;
					}
				}
				return torch::tensor({q1, q2, q3, qw}, pred.options());
			};

			auto findIdx = [&](const std::string& name) -> int {
				for (int i = 0; i < (int)RigDesc.Bindings.size(); ++i)
					if (RigDesc.Bindings[i].BoneName == name) return i;
				return -1;
			};

			int pelvisIdx  = findIdx("pelvis");
			int thighRIdx  = findIdx("thigh_r");
			int calfRIdx   = findIdx("calf_r");
			int thighLIdx  = findIdx("thigh_l");
			int calfLIdx   = findIdx("calf_l");

			if (pelvisIdx < 0 || thighRIdx < 0 || calfRIdx < 0 || thighLIdx < 0 || calfLIdx < 0)
			{
				std::cerr << "Missing pelvis, thigh_r, calf_r, thigh_l, or calf_l in rig description" << std::endl;
				return result;
			}

			// Pelvis loss
			auto pelvisPos  = pred.index({0, torch::indexing::Slice(0, 3)});
			auto pelvisQDelta = pred.index({0, torch::indexing::Slice(3, 7)});
			auto tPelvisPos   = target.index({0, torch::indexing::Slice(0, 3)});
			auto tPelvisQ     = target.index({0, torch::indexing::Slice(3, 7)});
			result.err_loss = result.err_loss + (torch::mse_loss(pelvisPos, tPelvisPos)) * 0.01f;
			result.err_loss = result.err_loss + (torch::mse_loss(pelvisQDelta, tPelvisQ)) * 0.01f;

			// World pelvis rotation matrix
			auto mPelvis = torch::mm(quatToMat(getBindQuat(pelvisIdx)), quatToMat(pelvisQDelta));

			// FK chain for one leg
			auto computeLegFKLoss = [&](int thighIdx, int calfIdx, int64_t footSliceStart) {
				const auto& tB = RigDesc.Bindings[thighIdx];
				const auto& cB = RigDesc.Bindings[calfIdx];

				// Hip socket = pelvis world pos + pelvis world rotation * thigh local offset
				// target[tB.Offset .. tB.Offset+2] = (offset_x, offset_y, offset_z) do thigh em local space da pelvis
				auto thighLocalOffset = target.index({0, torch::indexing::Slice(tB.Offset, tB.Offset + 3)}).detach();
				auto hipPos = pelvisPos + torch::mv(mPelvis, thighLocalOffset);

				// Thigh bone length vector in thigh local space
				// offset_z = t_offset = bone_l1, the bone length along local Z
				float boneL1z = target.index({0, tB.Offset + 2}).item<float>();
				auto vThigh = torch::tensor({0.f, 0.f, boneL1z}, pred.options());

				// mThigh = mPelvis * bindPose_thigh * delta_thigh  (pelvis propagates here)
				auto qThighDelta = pred.index({0, torch::indexing::Slice(tB.Offset + 3, tB.Offset + 7)});
				auto mThigh = torch::mm(mPelvis, torch::mm(quatToMat(getBindQuat(thighIdx)), quatToMat(qThighDelta)));
				auto kneePos = hipPos + torch::mv(mThigh, vThigh);

				// Calf bone length vector in calf local space
				float boneL2z = target.index({0, cB.Offset + 2}).item<float>();
				auto vCalf = torch::tensor({0.f, 0.f, boneL2z}, pred.options());

				// mCalf = mThigh * bindPose_calf * delta_calf  (thigh propagates here)
				auto qCalfDelta = pred.index({0, torch::indexing::Slice(cB.Offset + 3, cB.Offset + 7)});
				auto mCalf = torch::mm(mThigh, torch::mm(quatToMat(getBindQuat(calfIdx)), quatToMat(qCalfDelta)));
				auto footPos = kneePos + torch::mv(mCalf, vCalf);

				// FK loss: foot IK target predicted by the model must match FK chain result
				auto footIKTarget = pred.index({0, torch::indexing::Slice(footSliceStart, footSliceStart + 3)});
				
				// Ensure footPos has the same options as pred to avoid memory/device errors
				auto footPosFinal = footPos.to(pred.options());
				result.err_loss = result.err_loss + (torch::mse_loss(footIKTarget, footPosFinal) * 10.0f);
			};

			computeLegFKLoss(thighRIdx, calfRIdx, 7);   // foot_r at offset 7
			computeLegFKLoss(thighLIdx, calfLIdx, 14);  // foot_l at offset 14

			// Thigh Spacing Loss: The lateral distance between predicted thighs must not exceed bone_l0
			auto boneL0_tensor = RigDesc.GetInputBoneValue(input, "bone_l0");
			if (boneL0_tensor.defined()) {
				float boneL0 = boneL0_tensor.item<float>();
				auto thighRPosPred = pred.index({0, torch::indexing::Slice(RigDesc.Bindings[thighRIdx].Offset, RigDesc.Bindings[thighRIdx].Offset + 3)});
				auto thighLPosPred = pred.index({0, torch::indexing::Slice(RigDesc.Bindings[thighLIdx].Offset, RigDesc.Bindings[thighLIdx].Offset + 3)});

				// offset_y is the lateral axis (RightVector)
				auto lateralR = thighRPosPred.index({1});
				auto lateralL = thighLPosPred.index({1});
				auto lateralDist = torch::abs(lateralR - lateralL);
				auto spacingViolation = torch::relu(lateralDist - boneL0);
				result.err_loss = result.err_loss + (spacingViolation * 10.0f);

				// RightVector Verification: Coxa R deve ser negativa e Coxa L deve ser positiva (conforme dataset)
				// E elas não devem "cruzar" (R deve estar sempre à direita de L)
				auto crossViolation = torch::relu(lateralR - lateralL); // Se R for maior que L, cruzaram
				result.err_loss = result.err_loss + (crossViolation * 50.0f);

				// Longitudinal alignment (X): Coxas não devem ficar uma na frente da outra (colisão)
				auto longitudinalDist = torch::abs(thighRPosPred.index({0}) - thighLPosPred.index({0}));
				result.err_loss = result.err_loss + (longitudinalDist * 30.0f);

				// Vertical alignment (Z): Coxas devem estar niveladas no plano da pelvis
				auto verticalDist = torch::abs(thighRPosPred.index({2}) - thighLPosPred.index({2}));
				result.err_loss = result.err_loss + (verticalDist * 30.0f);
			}

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
