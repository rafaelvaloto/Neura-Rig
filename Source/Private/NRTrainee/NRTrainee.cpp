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
#include "NRTrainee/NRTrainee.h"

#include <ranges>

#include "NRCore/NRRules.h"
#include "NRCore/NRTypes.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, NRModelProfile Rig, NRRules& Ev, const double LearningRate)
		: TargetModel(TargetModel)
		  , Evaluator(Ev)
		  , RigDesc(std::move(Rig))
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));

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
					std::cerr << "[NRTrainee] Rule not found: " << RigDesc.Bindings[i].RuleName << std::endl;
					continue;
				}
				Evaluator.Setup(F_rule, i);
			}
		}
	}

	template<FloatingPoint T>
	float NRTrainee<T>::TrainStep(const std::vector<float>& InputFloats)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t BatchSize = static_cast<int32_t>(InputFloats.size()) / InCount;

		const auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
		auto InputTensor = torch::from_blob((void*)InputFloats.data(), {BatchSize, InCount}, options).clone();

		Optimizer->zero_grad();
		auto Prediction = TargetModel->Forward(InputTensor);
		auto [TotalLoss, PosLoss, RotationLoss, RegLoss, FKLoss] = ComputeRLReward(InputTensor, Prediction);

		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);

		Optimizer->step();
		return TotalLoss.template item<float>();
	}

	int frameCounter = 0;
	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& InputTensor, const torch::Tensor& Pred)
	{
		const auto T_ideal = torch::zeros_like(Pred);
		auto B_size = RigDesc.Bindings.size();
		for (auto i = 0; i < B_size; ++i)
		{
			auto& varMap = Evaluator.Parsers[i].GetVar();

			auto& Bindings = RigDesc.Bindings[i];
			int R_size = Bindings.Rules.size();
			for (auto j = 0; j < R_size; ++j)
			{
				auto& F_rule = Bindings.Rules[j];
				if (F_rule.Name.empty())
				{
					std::cerr << "[NRTrainee] Rule not found: " << RigDesc.Bindings[i].RuleName << std::endl;
					continue;
				}

				Evaluator.SetTensorInputs(i, F_rule, RigDesc, InputTensor);
				for (auto& [Name, Expr] : F_rule.Logic)
				{
					auto it = varMap.find(Name);
					if (it == varMap.end())
					{
						std::cerr << "[NRTrainee] Logic variable not found: " << Name << std::endl;
						continue;
					}

					*(it->second) = Evaluator.Eval(i, Expr);
				}

				for (auto& [Id, Condition, Formulas] : F_rule.Phases)
				{
					auto F_x = Id + "_condition";
					auto it = varMap.find(F_x);
					if (it == varMap.end())
					{
						std::cerr << "[NRTrainee] Phase condition variable not found: " << Id + "_condition" << std::endl;
						continue;
					}

					*(it->second) = Evaluator.Eval(i, Condition);
					if (*(it->second) == 0)
					{
						continue;
					}

					int B_idx = 0;
					for (auto& [formulaName, expr] : Formulas)
					{
						auto k = Id + "_" + formulaName;
						if (auto F_it = varMap.find(k); F_it != varMap.end())
						{
							*(F_it->second) = Evaluator.Eval(i, expr);
							T_ideal[0][Bindings.Offset + B_idx] = *(F_it->second);
						}

						B_idx++;
					}
					break;
				}
			}
		}

		std::vector<int64_t> ROffsets;
		ROffsets.reserve(2);
		std::vector<int64_t> LOffsets;
		LOffsets.reserve(2);
		for (size_t i = 0; i < B_size; ++i)
		{
			auto& Bindings = RigDesc.Bindings[i];

			if (Bindings.BoneName == "calf_r") ROffsets.push_back(Bindings.Offset + 3);
			else if (Bindings.BoneName == "thigh_r") ROffsets.push_back(Bindings.Offset + 3);
			else if (Bindings.BoneName == "calf_l") LOffsets.push_back(Bindings.Offset + 3);
			else if (Bindings.BoneName == "thigh_l") LOffsets.push_back(Bindings.Offset + 3);
		}

		Predicated = Pred;
		IdealTargets = T_ideal;

		auto P1_xyz = Pred.index({0, torch::indexing::Slice(0, 3)});
		auto P2_xyz = Pred.index({0, torch::indexing::Slice(6, 9)});
		auto T1_xyz = T_ideal.index({0, torch::indexing::Slice(0, 3)});
		auto T2_xyz = T_ideal.index({0, torch::indexing::Slice(6, 9)});


		auto FK = ValidateFeetFK(Pred, LOffsets, ROffsets, false);
		const auto Pos1Loss = torch::mse_loss(P1_xyz, T1_xyz);
		const auto Pos2Loss = torch::mse_loss(P2_xyz, T2_xyz);

		// Loss total
		const auto P1Loss = Pos1Loss + Pos2Loss *1.0;
		const auto TotalLoss = P1Loss + (FK.err_loss * 5.5);

		if (frameCounter++ % 30 == 0)
		{
			std::cout << "Frame: " << frameCounter << std::endl;
			std::cout << "Total: " << TotalLoss.item<float>() << std::endl;
		}
		return IKLossResult(TotalLoss, TotalLoss, TotalLoss, TotalLoss, FK.err_loss);
	}


	template<FloatingPoint T>
	void NRTrainee<T>::SaveWeights(const std::string& Path)
	{
		torch::save(TargetModel, Path);
	}

	template<FloatingPoint T>
	void NRTrainee<T>::LoadWeights(const std::string& Path)
	{
		torch::load(TargetModel, Path);
		TargetModel->train();
	}

	template<FloatingPoint T>
	void NRTrainee<T>::Reset()
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
		for (auto& deq : PredXHistory)
			deq.clear();
		for (auto& deq : IdealXHistory)
			deq.clear();

		IdealTargets = torch::zeros({1, RigDesc.GetRequiredOutputSize()});
		Predicated = torch::zeros({1, RigDesc.GetRequiredOutputSize()});
		std::cout << "[NRTrainee] Reset complete. Clean state for training." << std::endl;
	}


	template class NRTrainee<float>;
	template class NRTrainee<double>;
} // namespace NR
