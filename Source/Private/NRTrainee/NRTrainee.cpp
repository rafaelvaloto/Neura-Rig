// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

#include "NRCore/NRRules.h"
#include "NRCore/NRTypes.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, NRModelProfile Rig, NRRules Ev, const double LearningRate)
		: TargetModel(TargetModel)
		  , RigDesc(Rig)
		  , Evaluator(Ev)
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));

		PredXHistory.resize(RigDesc.Bindings.size());
		IdealXHistory.resize(RigDesc.Bindings.size());

		for (int i = 0; i < RigDesc.Bindings.size(); ++i)
		{
			auto& binding = RigDesc.Bindings[i];

			auto rule = RigDesc.FindRule(binding.RuleName);
			Evaluator.Setup(rule, i);
		}
	}

	template<FloatingPoint T>
	float NRTrainee<T>::TrainStep(const std::vector<float>& InputFloats)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t BatchSize = static_cast<int32_t>(InputFloats.size()) / InCount;

		auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
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
		const int64_t BatchSize = Input.size(0);
		auto IdealTargets = torch::zeros_like(Pred);

		Evaluator.ResetTime();
		for (int64_t sample = 0; sample < BatchSize; ++sample)
		{
			for (int i = 0; i < RigDesc.Bindings.size(); ++i)
			{
				auto& binding = RigDesc.Bindings[i];
				auto rule = RigDesc.FindRule(binding.RuleName);
				Evaluator.SetTensorInputs(i, rule, RigDesc, Input);

				for (auto const& [name, expr] : rule.Logic)
				{
					Evaluator.Vars[i][name] = Evaluator.Eval(i, expr);
				}

				for (auto const& phase : rule.Phases)
				{
					const double sign = Evaluator.Eval(i, phase.Condition);
					// std::cout << "==========P=========" << std::endl;
					// std::cout << "BIND " << i
					// 	<< " ID=" << phase.Id
					// 	<< " Bone=" << binding.BoneName
					// 	<< " Rule=" << binding.RuleName
					// 	<< " Condition=" << sign
					// 	<< " bone_l1=" << Evaluator.Vars[i].at("bone_l1")
					// 	<< " bone_l2=" << Evaluator.Vars[i].at("bone_l2")
					// 	<< " velocity=" << Evaluator.Vars[i].at("velocity")
					// 	<< " t_cycle=" << Evaluator.Vars[i].at("t_cycle")
					// 	<< " cycle=" << Evaluator.Vars[i].at("cycle")
					// 	<< " stride_amp=" << Evaluator.Vars[i].at("stride_amp")
					// 	<< std::endl;
					// std::cout << "==========P=========" << std::endl;

					if (sign == 0)
					{
						continue;
					}

					int idx = 0;
					for (auto const& [fname, fexpr] : phase.Formulas)
					{
						double value = Evaluator.Eval(i, fexpr);
						if (!std::isfinite(value))
						{
							std::cerr << "[NRTrainee] Non-finite formula value for [" << fname
								<< "] expr=[" << fexpr << "]" << std::endl;
							value = 0.0;
						}

						if (value == 0.0)
						{
							idx++;
							continue;
						}

						Evaluator.Vars[i][fname] = value;
						IdealTargets[sample][0] = static_cast<float>(value);
						IdealTargets[sample][3] = -static_cast<float>(value);

						IdealTarg = IdealTargets;
						break;
					}
					break;
				}
			}
		}

		auto PredX = torch::stack({Pred.index({0, 0}), Pred.index({0, 3})});
		auto TargetX = torch::stack({IdealTargets.index({0, 0}), IdealTargets.index({0, 3})});
		auto PosLoss = torch::mse_loss(PredX, TargetX);

		auto diff = Pred.index({0, 0}) - Pred.index({0, 3});
		auto ideal= IdealTargets.index({0, 0}) - IdealTargets.index({0, 3});
		auto AmpLoss = torch::mse_loss(diff, ideal);

		auto TotalLoss = (PosLoss * 1.0f) + (AmpLoss * 5.0f);

		auto Zero = torch::zeros({0}, torch::kFloat);
		return IKLossResult(TotalLoss, Zero, Zero, Zero);
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


	template class NRTrainee<float>;
	template class NRTrainee<double>;
} // namespace NR
