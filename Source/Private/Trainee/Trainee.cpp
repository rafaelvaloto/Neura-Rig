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
#include "Trainee/Trainee.h"

#include <ranges>

#include "Core/Rules.h"
#include "Core/Types.h"

namespace NR
{
	template<FloatingPoint T>
	Trainee<T>::Trainee(std::shared_ptr<IModel<T> > TargetModel, NRModelProfile Rig, Rules& Ev, const double LearningRate)
		: TargetModel(TargetModel)
		  , Evaluator(Ev)
		  , RigDesc(std::move(Rig))
	{
		double finalLR = LearningRate;
		if (RigDesc.TrainingWeights.HyperParameters.LearningRate > 0)
		{
			finalLR = RigDesc.TrainingWeights.HyperParameters.LearningRate;
		}

		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(finalLR));

		auto T_size = RigDesc.GetRequiredOutputSize();
		IdealTargets = torch::zeros_like(torch::empty({1, T_size}));
		Predicated = torch::zeros_like(torch::empty({1, T_size}));

		auto B_size = RigDesc.Bindings.size();
		for (auto i = 0; i < B_size; ++i)
		{
			int R_size = RigDesc.Bindings[i].Rules.size();
			for (auto j = 0; j < R_size; ++j)
			{
				auto& F_rule = RigDesc.Bindings[i].Rules[j];
				if (F_rule.Name.empty())
				{
					std::cerr << "[Trainee] Rule not found: " << RigDesc.Bindings[i].RuleName << std::endl;
					continue;
				}
				Evaluator.Setup(F_rule, i);
			}
		}
	}

	template<FloatingPoint T>
	torch::Tensor Trainee<T>::ChooseBestPrediction(
		const std::vector<torch::Tensor>& candidates,
		const torch::Tensor& target,
		const torch::Tensor& prevPred)
	{
		if (candidates.empty())
		{
			return torch::zeros_like(target);
		}

		float bestScore = std::numeric_limits<float>::max();
		torch::Tensor best = candidates.front();

		for (const auto& pred : candidates)
		{
			auto posLoss = torch::mse_loss(pred, target).template item<float>();
			float tempLoss = 0.0f;
			if (prevPred.defined() && prevPred.numel() > 0)
			{
				tempLoss = torch::mse_loss(pred, prevPred).template item<float>();
			}

			float score = posLoss + 0.35f * tempLoss;

			if (score < bestScore)
			{
				bestScore = score;
				best = pred;
			}
		}

		return best;
	}

	template<FloatingPoint T>
	float Trainee<T>::TrainStep(const std::vector<float>& InputFloats)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t BatchSize = static_cast<int32_t>(InputFloats.size()) / InCount;

		const auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
		auto InputTensor = torch::from_blob((void*)InputFloats.data(), {BatchSize, InCount}, options).clone();

		Optimizer->zero_grad();
		auto Prediction = TargetModel->Forward(InputTensor);

		// Calculate Ideal Target for this input
		auto T_ideal = torch::zeros_like(Prediction);
		for (size_t i = 0; i < RigDesc.Bindings.size(); ++i)
		{
			auto& varMap = Evaluator.Parsers[i].GetVar();
			auto& binding = RigDesc.Bindings[i];

			for (const auto& F_rule : binding.Rules)
			{
				if (F_rule.Logic.empty())
					continue;

				Evaluator.SetTensorInputs(i, F_rule, RigDesc, InputTensor);
				for (const auto& [Name, Expr] : F_rule.Logic)
				{
					if (auto it = varMap.find(Name); it != varMap.end())
						*(it->second) = Evaluator.Eval(i, Expr);
				}

				for (const auto& [Id, Condition, Formulas] : F_rule.Phases)
				{
					if (Evaluator.Eval(i, Condition) == 0)
						continue;

					int B_idx = 0;
					for (const auto& [formulaName, expr] : Formulas)
					{
						T_ideal[0][binding.Offset + B_idx] = Evaluator.Eval(i, expr);
						++B_idx;
					}
					break;
				}
			}
		}

		auto Result = ComputeLoss(Prediction, T_ideal, InputTensor, PredHistory);

		Result.TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(true), 1.0);

		Optimizer->step();

		// Update histories and internal states
		PredHistory2 = PredHistory.defined() ? PredHistory.detach().clone() : torch::Tensor();
		PredHistory = Prediction.detach().clone();
		PredictionCandidates.push_back(Prediction.detach().clone());
		if (PredictionCandidates.size() > MaxCandidates)
			PredictionCandidates.erase(PredictionCandidates.begin());

		float emaAlpha = RigDesc.TrainingWeights.HyperParameters.EmaAlpha;
		if (!SmoothedOutput.defined())
			SmoothedOutput = Prediction.detach().clone();
		else
			SmoothedOutput = SmoothedOutput * (1.0f - emaAlpha) + Prediction.detach() * emaAlpha;

		Predicated = Prediction;
		IdealTargets = T_ideal;

		return Result.TotalLoss.template item<float>();
	}

	template<FloatingPoint T>
	IKLossResult Trainee<T>::ComputeLoss(const torch::Tensor& Pred, const torch::Tensor& Target, const torch::Tensor& Input, const torch::Tensor& PrevPred)
	{
		const auto& TW = RigDesc.TrainingWeights;
		IKLossResult res;
		res.TotalLoss = torch::tensor(0.0f, Pred.options());

		auto getWeight = [&](const std::string& name) {
			auto it = TW.LossWeights.find(name);
			return (it != TW.LossWeights.end()) ? it->second.Weight : 1.0f;
		};

		auto Target_r = Target.index({0, torch::indexing::Slice(21, 24)});
		auto Target_l = Target.index({0, torch::indexing::Slice(42, 45)});
		auto Pre_r = Pred.index({0, torch::indexing::Slice(21, 24)});
		auto Pre_l = Pred.index({0, torch::indexing::Slice(42, 45)});

		// Quaternion Norm Loss
		res.QuaternionNormLoss = torch::tensor(0.0f, Pred.options());
		res.RotationLoss = torch::tensor(0.0f, Pred.options());

		// 1. Kinematics Loss (FK)
		res.KinematicsLoss = ComputeFK(Pred, Target);
		res.TotalLoss = res.TotalLoss + res.KinematicsLoss * getWeight("Objective");

		// 2. Position Loss (MSE between Pred and Ideal Target)
		res.PositionLoss = torch::mse_loss(Pre_r, Target_r) * getWeight("Position");
		res.PositionLoss = res.PositionLoss + torch::mse_loss(Pre_l, Target_l) * getWeight("Position");
		res.TotalLoss = res.TotalLoss + res.PositionLoss;

		// 3. Temporal Loss
		if (PrevPred.defined() && PrevPred.numel() > 0)
		{
			res.TemporalLoss = torch::mse_loss(Pred, PrevPred);
			res.TotalLoss = res.TotalLoss + res.TemporalLoss * getWeight("Temporal");
		}
		else
		{
			res.TemporalLoss = torch::tensor(0.0f, Pred.options());
		}

		// 4. Acceleration Loss
		if (PredHistory2.defined() && PredHistory2.numel() > 0)
		{
			auto velNow = (Pred - PrevPred);
			auto velPrev = (PrevPred - PredHistory2);
			res.AccelerationLoss = torch::mse_loss(velNow, velPrev);
			res.TotalLoss = res.TotalLoss + res.AccelerationLoss * getWeight("Acceleration");
		}
		else
		{
			res.AccelerationLoss = torch::tensor(0.0f, Pred.options());
		}

		// 5. Smooth Output Loss (Delta to EMA)
		if (SmoothedOutput.defined())
		{
			res.SmoothOutputLoss = torch::mse_loss(Pred, SmoothedOutput);
			res.TotalLoss = res.TotalLoss + res.SmoothOutputLoss * getWeight("SmoothOutput");
		}
		else
		{
			res.SmoothOutputLoss = torch::tensor(0.0f, Pred.options());
		}

		auto computeBoneLimitsLoss = [&](const NRSkeleton::Bone& bone) {
			auto qDelta = Pred.index({0, torch::indexing::Slice(bone.Offset + 3, bone.Offset + 7)});

			float minXR = DegToRad(bone.Limits.MinX);
			float maxXR = DegToRad(bone.Limits.MaxX);
			float minYR = DegToRad(bone.Limits.MinY);
			float maxYR = DegToRad(bone.Limits.MaxY);
			float minZR = DegToRad(bone.Limits.MinZ);
			float maxZR = DegToRad(bone.Limits.MaxZ);

			auto ex = 2.0f * qDelta[0];
			auto ey = 2.0f * qDelta[1];
			auto ez = 2.0f * qDelta[2];

			float wIk = getWeight("Kinematics");

			//
			auto lx = wIk * torch::pow(torch::clamp(ex - maxXR, 0.0f) + torch::clamp(minXR - ex, 0.0f), 2);
			auto ly = wIk * torch::pow(torch::clamp(ey - maxYR, 0.0f) + torch::clamp(minYR - ey, 0.0f), 2);
			auto lz = wIk * torch::pow(torch::clamp(ez - maxZR, 0.0f) + torch::clamp(minZR - ez, 0.0f), 2);

			return lx + ly + lz;
		};

		// Apply parent limits
		torch::Tensor limitsLoss = torch::tensor(0.0f, Pred.options());
		limitsLoss = limitsLoss + computeBoneLimitsLoss(RigDesc.Skeleton.Parent);
		res.TotalLoss = res.TotalLoss + limitsLoss;

		for (const auto& binding : RigDesc.Bindings)
		{
			if (binding.Size >= 7)
			{
				auto q = Pred.index({0, torch::indexing::Slice(binding.Offset + 3, binding.Offset + 7)});
				res.QuaternionNormLoss = res.QuaternionNormLoss + torch::pow(torch::norm(q, 2) - 1.0f, 2);

				// Apply bone limits
				for (const auto& chain : RigDesc.Skeleton.Rest)
				{
					for (const auto& bone : chain)
					{
						if (bone.Name == binding.BoneName)
						{
							limitsLoss = limitsLoss + computeBoneLimitsLoss(bone);
						}
					}
				}
			}
		}
		limitsLoss = limitsLoss * getWeight("Kinematics");
		res.TotalLoss = res.TotalLoss + res.QuaternionNormLoss * getWeight("QuaternionNorm");

		static int frameCounter = 0;
		if (++frameCounter % 30 == 0)
		{
			std::cout << "--- Frame: " << frameCounter << " ---" << std::endl;
			std::cout << "  PosLoss: " << res.PositionLoss.item<float>() << " * " << getWeight("Position") << std::endl;
			std::cout << "  KinLoss: " << res.KinematicsLoss.item<float>() << " * " << getWeight("Kinematics") << std::endl;
			std::cout << "  TempLoss: " << res.TemporalLoss.item<float>() << " * " << getWeight("Temporal") << std::endl;
			std::cout << "  AccLoss: " << res.AccelerationLoss.item<float>() << " * " << getWeight("Acceleration") << std::endl;
			std::cout << "  SmoothLoss: " << res.SmoothOutputLoss.item<float>() << " * " << getWeight("SmoothOutput") << std::endl;
			std::cout << "  QuatLoss: " << res.QuaternionNormLoss.item<float>() << " * " << getWeight("QuaternionNorm") << std::endl;
			std::cout << "  LimitLoss: " << limitsLoss.item<float>() << " * 20.0" << std::endl;
			std::cout << "  Total: " << res.TotalLoss.item<float>() << std::endl;
			std::cout << "-----------------------" << std::endl;
		}

		return res;
	}

	template<FloatingPoint T>
	torch::Tensor Trainee<T>::ComputeFK(const torch::Tensor& Pred, const torch::Tensor& Target)
	{
		const auto& SK = RigDesc.Skeleton;
		auto totalLoss = torch::zeros({}, Pred.options());

		auto quatToMat = [&](const torch::Tensor& q) -> torch::Tensor {
			auto qn = q / (q.norm(2, 0) + 1e-8);

			auto x = qn[0];
			auto y = qn[1];
			auto z = qn[2];
			auto w = qn[3];

			auto row1 = torch::stack({1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y});
			auto row2 = torch::stack({2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x});
			auto row3 = torch::stack({2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y});

			return torch::stack({row1, row2, row3});
		};

		auto pParent = Pred.index({0, torch::indexing::Slice(SK.Parent.Offset, SK.Parent.Offset + 3)});
		auto qParent = Pred.index({0, torch::indexing::Slice(SK.Parent.Offset + 3, SK.Parent.Offset + 7)});
		auto mParentPred = quatToMat(qParent);

		auto pParentRest = torch::tensor({SK.Parent.RestPose.x, SK.Parent.RestPose.y, SK.Parent.RestPose.z}, Pred.options());
		auto qParentRest = torch::tensor({SK.Parent.RestPose.q1,SK.Parent.RestPose.q2,SK.Parent.RestPose.q3,SK.Parent.RestPose.qw}, Pred.options());
		auto mParentRest = quatToMat(qParentRest);

		const auto& TW = RigDesc.TrainingWeights;
		float wFk = TW.LossWeights.at("Kinematics").Weight;
		float wQuat = TW.LossWeights.at("QuaternionNorm").Weight;

		for (size_t chainIdx = 0; chainIdx < SK.Rest.size(); ++chainIdx)
		{
			auto pChain = pParent;
			auto mChain = torch::mm(mParentPred, mParentRest);

			std::cout << "=============" << SK.Parent.Name << "=============" << std::endl;
			std::cout << " pParentRest: " << pParentRest << std::endl;
			std::cout << " qParentRest: " << qParentRest << std::endl;
			std::cout << " mParentRest: " << mParentRest << std::endl;
			std::cout << "===================================================" << std::endl;

			for (size_t boneIdx = 0; boneIdx < SK.Rest[chainIdx].size(); ++boneIdx)
			{
				const auto& bone = SK.Rest[chainIdx][boneIdx];

				// Rest bones pose
				auto pRest = torch::tensor(
					{bone.RestPose.x, bone.RestPose.y, bone.RestPose.z},
					Pred.options()
				);
				auto qRest = torch::tensor({bone.RestPose.q1,bone.RestPose.q2,bone.RestPose.q3,bone.RestPose.qw}, Pred.options());
				auto mRest = quatToMat(qRest);

				// Predicate bones pose
				auto pLocal = Pred.index({0, torch::indexing::Slice(bone.Offset, bone.Offset + 3)});
				auto qLocal = Pred.index({0, torch::indexing::Slice(bone.Offset + 3, bone.Offset + 7)});
				auto mLocal = quatToMat(qLocal);

				// pChain + mChain * pRest
				auto pChild = pChain + torch::mv(mChain, (pLocal - pRest));
				auto mChild = torch::mm(mChain, (mLocal * mRest));

				// Bias Multiplier
				float bonePosMultiplier = 1.0f;
				for (const auto& bias : TW.BoneSpecificBias)
				{
					if (bias.Name == bone.Name)
					{
						bonePosMultiplier = bias.PositionMultiplier;
						break;
					}
				}

				auto predLength = torch::norm(pLocal, 2);
				auto restLength = torch::norm(pRest, 2);

				// Penalty for bone length
				auto boneLengthLoss = torch::pow(predLength - restLength, 2) * bonePosMultiplier;

				// Penalty for quaternion normalization
				auto quatNormLoss = torch::pow(torch::norm(qLocal, 2) - 1.0f, 2) * wQuat;

				std::cout << "=============" << bone.Name << "=============" << std::endl;
				std::cout << " mParentPred: " << mParentPred << " mParentRest: " << mParentRest << std::endl;
				std::cout << " pChain: " << pChain << " mChain: " << mChain << std::endl;
				std::cout << " Length: " << predLength << " Rest: " << restLength << std::endl;
				std::cout << "===================================================" << std::endl;

				totalLoss = totalLoss + quatNormLoss + boneLengthLoss;

				if (boneIdx == SK.Rest[chainIdx].size() - 1)
				{
					auto pTarget = Target.index({0, torch::indexing::Slice(bone.Offset, bone.Offset + 3)});
					std::cout << "=============" << bone.Name << "=============" << std::endl;
					std::cout << " pChild: " << pChild << std::endl;
					std::cout << " pTarget:" << pTarget << std::endl;
					std::cout << "===================================================" << std::endl;

					totalLoss = totalLoss + (torch::mse_loss(pChild, pTarget) * wFk);
				}

				pChain = pChild;
				mChain = mChild;
			}
		}

		return totalLoss;
	}


	template<FloatingPoint T>
	void Trainee<T>::SaveWeights(const std::string& Path)
	{
		torch::save(TargetModel, Path);
	}

	template<FloatingPoint T>
	void Trainee<T>::LoadWeights(const std::string& Path)
	{
		torch::load(TargetModel, Path);
		TargetModel->train();
	}

	template<FloatingPoint T>
	void Trainee<T>::Reset()
	{
		for (auto& p : TargetModel->parameters())
		{
			if (p.dim() >= 2)
				torch::nn::init::xavier_uniform_(p);
			else
				torch::nn::init::zeros_(p);
		}

		double lr = Optimizer->param_groups()[0].options().get_lr();
		Optimizer = std::make_unique<torch::optim::Adam>(
			TargetModel->parameters(),
			torch::optim::AdamOptions(lr)
			);

		Evaluator.deltaTime = 0.0;
		PredictionCandidates.clear();
		for (auto& deq : PredXHistory)
			deq.clear();
		for (auto& deq : IdealXHistory)
			deq.clear();

		IdealTargets = torch::zeros({1, RigDesc.GetRequiredOutputSize()});
		Predicated = torch::zeros({1, RigDesc.GetRequiredOutputSize()});
		std::cout << "[Trainee] Reset complete. Clean state for training." << std::endl;
	}


	template class Trainee<float>;
	template class Trainee<double>;
} // namespace NR
