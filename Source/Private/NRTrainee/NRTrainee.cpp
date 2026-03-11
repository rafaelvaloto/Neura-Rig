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

		auto size = RigDesc.GetRequiredOutputSize();
		IdealTargets = torch::zeros_like(torch::empty({1, size}));
		Predicated = torch::zeros_like(torch::empty({1, size}));

		const int b_size = RigDesc.Bindings.size();
		V_rules.reserve(b_size);
		for (int i = 0; i < RigDesc.Bindings.size(); ++i)
		{
			auto& [BoneName, RuleName, Size, Offset] = RigDesc.Bindings[i];
			auto F_rule = RigDesc.FindRule(RuleName);
			Evaluator.Setup(F_rule, i);

			if (V_rules.find(F_rule.Name) != V_rules.end())
			{
				continue;
			}
			V_rules.emplace(F_rule.Name, F_rule);
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
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& Input, const torch::Tensor& Pred)
	{
		auto T_pred = Pred;
		auto T_ideal = torch::zeros_like(Pred);

		int b_size = RigDesc.Bindings.size();
		for (auto rule : V_rules | std::views::values)
		{
			for (int i = 0; i < b_size; ++i)
			{
				Evaluator.SetTensorInputs(i, rule, RigDesc, Input);
				for (auto const& [name, expr] : rule.Logic)
				{
					Evaluator.Vars[i][name] = Evaluator.Eval(i, expr);
				}

				for (const auto& [Id, Condition, Formulas] : rule.Phases)
				{
					if (Evaluator.Eval(i, Condition) > 0)
					{
						int S_idx = 0;
						auto R_idx = RigDesc.Bindings[i];
						std::cout << "[NRTrainee] Evaluating " << Id <<  " in " << R_idx.BoneName <<  std::endl;

						for (auto const& [fname, fexpr] : Formulas)
						{
							auto val = Evaluator.Eval(i, fexpr);
							Evaluator.Vars[i][fname] = val;
							T_ideal[0][R_idx.Offset + S_idx] = val;

							std::cout << "[NRTrainee] IdealTargets[" << R_idx.Offset + S_idx << "] : " << IdealTargets[0][R_idx.Offset + S_idx] << std::endl;
							S_idx++;
						}
						break;
					}
				}
			}
		}

		Predicated = T_pred;
		IdealTargets = T_ideal;

		const auto PosLoss = torch::mse_loss(T_pred, T_ideal);
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
		for (auto& varMap : Evaluator.Vars)
		{
			for (auto& [key, val] : varMap)
			{
				val = 0.0;
			}
		}

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
