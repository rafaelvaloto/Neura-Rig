// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

#include <ranges>

#include "NRCore/NRRules.h"
#include "NRCore/NRTypes.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, const NRModelProfile& Rig, const NRRules& Ev, const double LearningRate)
		: TargetModel(TargetModel)
		  , Evaluator(Ev)
		  , RigDesc(Rig)
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
		auto [TotalLoss, PosLoss, RotationLoss, RegLoss] = ComputeRLReward(InputTensor, Prediction);

		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);

		Optimizer->step();
		return TotalLoss.template item<float>();
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& InputTensor, const  torch::Tensor& Pred)
	{
		auto T_ideal = torch::zeros_like(Pred);
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
				for (auto& logic : F_rule.Logic)
				{
					auto it = varMap.find(logic.Name);
					if (it == varMap.end()) {
						std::cerr << "[NRTrainee] Logic variable not found: " << logic.Name << std::endl;
						continue;
					}

					*(it->second) = Evaluator.Eval(i, logic.Expr);
				}

				for (auto& [Id, Condition, Formulas] : F_rule.Phases)
				{
					auto it = varMap.find(Id + "_condition");
					if (it == varMap.end()) {
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
						auto F_it = varMap.find(Id + "_" + formulaName);
						if (F_it != varMap.end())
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

		Predicated = Pred;
		IdealTargets = T_ideal;

		const auto PosLoss = torch::mse_loss(Pred, T_ideal);
		const auto TotalLoss = (PosLoss* 1.0f);
		return IKLossResult(TotalLoss, PosLoss, PosLoss, PosLoss);
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

		IdealTargets = torch::Tensor();
		std::cout << "[NRTrainee] Reset complete. Clean state for training." << std::endl;
	}


	template class NRTrainee<float>;
	template class NRTrainee<double>;
} // namespace NR
