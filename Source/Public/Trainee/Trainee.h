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

		[[nodiscard]] FootValidationPair ValidateFeetFK(
			const torch::Tensor& pred,
			const torch::Tensor& target,
			const std::vector<int64_t>& rightOffsets,
			const std::vector<int64_t>& leftOffsets,
			bool anglesAreDegrees = true)
		{
			FootValidationPair result;
			result.err_loss = torch::tensor(0.0f, pred.options());

			auto getInitialQuat = [&](int bindingIdx) -> torch::Tensor {
				if (bindingIdx >= 0 && bindingIdx < RigDesc.Bindings.size()) {
					const auto& binding = RigDesc.Bindings[bindingIdx];
					if (!binding.Rules.empty() && !binding.Rules[0].Phases.empty()) {
						const auto& phase = binding.Rules[0].Phases[0];

						auto findVal = [&](const std::string& name) -> float {
							for(const auto& formula : phase.Formulas) {
								if(formula.Name == name) return std::stof(formula.Expr);
							}
							return 0.0f;
						};

						float q1 = findVal("q1");
						float q2 = findVal("q2");
						float q3 = findVal("q3");
						float qw = findVal("qw");
						if (qw == 0.0f && q1 == 0.0f && q2 == 0.0f && q3 == 0.0f) qw = 1.0f;

						return torch::tensor({q1, q2, q3, qw}, pred.options());
					}
				}
				return torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, pred.options());
			};


			auto findBindingIdx = [&](const std::string& name) -> int {
				for (int i = 0; i < (int)RigDesc.Bindings.size(); ++i)
				{
					if (RigDesc.Bindings[i].BoneName == name)
						return i;
				}
				return -1;
			};

			int pelvisBindingIdx = findBindingIdx("pelvis");

			auto computeFoot = [&](const std::vector<int64_t>& rotationOffsets, const torch::Tensor& footIKOffset, bool isRightLeg, int thighBindingIdx, int calfBindingIdx) {
				if (rotationOffsets.size() < 2 || thighBindingIdx == -1 || calfBindingIdx == -1)
					return;

				const auto& thighBinding = RigDesc.Bindings[thighBindingIdx];
				const auto& calfBinding = RigDesc.Bindings[calfBindingIdx];

				auto getQuatRotMat = [&](torch::Tensor q) {
					q = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
					auto x = q[0], y = q[1], z = q[2], w = q[3];
					return torch::stack({
						torch::stack({1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y}),
						torch::stack({2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x}),
						torch::stack({2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y})
					});
				};

				auto applyRightLegYawOffset = [&](torch::Tensor q) {
					if (!isRightLeg) return q;
					// O usuário indicou que apenas o Yaw (Z) da perna direita precisa do offset de -180.
					// No Unreal (Control Rig), multiplicar por (Pitch=0, Yaw=-180, Roll=0) rotaciona 180 graus no eixo Z.
					// Quat de (0,0,-180) em Euler -> (X=0, Y=0, Z=sin(-90), W=cos(-90)) = (0, 0, -1, 0)
					
					auto q_offset = torch::tensor({0.0f, 0.0f, -1.0f, 0.0f}, q.options());
					
					auto w1 = q_offset[3], x1 = q_offset[0], y1 = q_offset[1], z1 = q_offset[2];
					auto w2 = q[3], x2 = q[0], y2 = q[1], z2 = q[2];
					
					auto rw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
					auto rx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
					auto ry = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
					auto rz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
					
					return torch::stack({rx, ry, rz, rw});
				};

				auto readQuatWithLimit = [&](int64_t offset, int bindingIdx) -> torch::Tensor {
					if (offset + 4 > pred.size(1))
						return torch::tensor({0.0f, 0.0f, 0.0f, 1.0f}, pred.options());

					auto q_raw = pred.index({0, torch::indexing::Slice(offset, offset + 4)}).clone();
					auto q = torch::nn::functional::normalize(q_raw, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

					if (bindingIdx >= 0 && bindingIdx < RigDesc.Bindings.size()) {
						const auto& binding = RigDesc.Bindings[bindingIdx];
						if (!binding.Rules.empty() && !binding.Rules[0].Limits.IsZero()) {
							// Se for perna direita, aplicamos o offset de Yaw para bater com os limites do JSON (se necessário)
							// Mas agora que temos bases reais, talvez o offset de -180 não seja mais necessário se a base já o contiver.
							// Mantenho por segurança se o usuário ainda vir pernas cruzadas.
							auto q_lim = (isRightLeg && binding.BoneName.find("pelvis") == std::string::npos) ? applyRightLegYawOffset(q) : q;
							
							auto x = q_lim[0], y = q_lim[1], z = q_lim[2], w = q_lim[3];
							auto sinr_cosp = 2 * (w * x + y * z);
							auto cosr_cosp = 1 - 2 * (x * x + y * y);
							auto roll = torch::atan2(sinr_cosp, cosr_cosp);

							auto sinp = 2 * (w * y - z * x);
							auto pitch = torch::asin(torch::clamp(sinp, -0.99999f, 0.99999f));

							auto siny_cosp = 2 * (w * z + x * y);
							auto cosy_cosp = 1 - 2 * (y * y + z * z);
							auto yaw = torch::atan2(siny_cosp, cosy_cosp);

							auto euler = torch::stack({roll, pitch, yaw}) * (180.0f / 3.1415926535f);
							
							auto normalize_angle = [](torch::Tensor angle) {
								return torch::atan2(torch::sin(angle * (3.1415926535f / 180.0f)), torch::cos(angle * (3.1415926535f / 180.0f))) * (180.0f / 3.1415926535f);
							};

							auto roll_norm  = normalize_angle(euler[0]);
							auto pitch_norm = normalize_angle(euler[1]);
							auto yaw_val    = normalize_angle(euler[2]);

							const auto& limits = binding.Rules[0].Limits;

							auto minB = torch::tensor({limits.MinX, limits.MinY, limits.MinZ}, q.options());
							auto maxB = torch::tensor({limits.MaxX, limits.MaxY, limits.MaxZ}, q.options());

							auto euler_to_check = torch::stack({roll_norm, pitch_norm, yaw_val});
							
							auto diff = torch::clamp(minB - euler_to_check, 0.0f).pow(2).sum() + torch::clamp(euler_to_check - maxB, 0.0f).pow(2).sum();
							result.err_loss = result.err_loss + (diff * 15.0f);
						}
					}
					return q;
				};

				auto mBaseThigh = getQuatRotMat(getInitialQuat(thighBindingIdx));
				auto mBaseCalf  = getQuatRotMat(getInitialQuat(calfBindingIdx));

				auto qDeltaThigh = readQuatWithLimit(rotationOffsets[1], thighBindingIdx);
				auto qDeltaCalf  = readQuatWithLimit(rotationOffsets[0], calfBindingIdx);

				auto mDeltaThigh = getQuatRotMat(qDeltaThigh);
				auto mDeltaCalf  = getQuatRotMat(qDeltaCalf);

				float L1 = target.index({0, torch::indexing::Slice(thighBinding.Offset, thighBinding.Offset + 3)}).norm().item<float>();
				float L2 = target.index({0, torch::indexing::Slice(calfBinding.Offset, calfBinding.Offset + 3)}).norm().item<float>();

				auto pelvisOffset = pred.index({0, torch::indexing::Slice(0, 3)});
				auto qPelvisDelta = readQuatWithLimit(3, pelvisBindingIdx);

				auto T_pelvisOffsetBase = target.index({0, torch::indexing::Slice(0, 3)});
				auto T_qPelvisBase = target.index({0, torch::indexing::Slice(3, 7)});

				result.err_loss = result.err_loss + torch::mse_loss(pelvisOffset, T_pelvisOffsetBase) * 1.0f;
				
				auto qPelvisBase = getInitialQuat(pelvisBindingIdx);
				auto mPelvisBase = getQuatRotMat(qPelvisBase);
				auto mPelvisDelta = getQuatRotMat(qPelvisDelta);
				auto mPelvis = torch::mm(mPelvisBase, mPelvisDelta);

				auto hipPos = pelvisOffset;
				auto mThigh = torch::mm(mPelvis, torch::mm(mBaseThigh, mDeltaThigh));
				auto vThigh = torch::tensor({0.0f, 0.0f, L1}, pred.options()).unsqueeze(1);
				auto kneePos = hipPos + torch::mm(mThigh, vThigh).squeeze(1);

				auto mCalf = torch::mm(mThigh, torch::mm(mBaseCalf, mDeltaCalf));
				auto vCalf = torch::tensor({0.0f, 0.0f, L2}, pred.options()).unsqueeze(1);
				auto footPos = kneePos + torch::mm(mCalf, vCalf).squeeze(1);

				auto posErrTensor = torch::mse_loss(footIKOffset, (footPos - hipPos));
				result.err_loss = result.err_loss + posErrTensor;
			};

			auto P1_offset = pred.index({0, torch::indexing::Slice(7, 10)});
			auto P2_offset = pred.index({0, torch::indexing::Slice(14, 17)});

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
