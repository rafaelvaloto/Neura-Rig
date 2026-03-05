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

		return TotalLoss.item<float>();
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& Input, const torch::Tensor& Pred)
	{
		int64_t batchSize = Input.size(0);
		auto IdealTargets = torch::zeros_like(Pred);

		for (int64_t b = 0; b < batchSize; ++b)
		{
			torch::Tensor currentInput = Input[b];

			for (int i = 0; i < RigDesc.Bindings.size(); ++i)
			{
				auto& binding = RigDesc.Bindings[i];
				auto rule = RigDesc.FindRule(binding.RuleName);

				Evaluator.SetTensorInputs(i, rule, RigDesc, currentInput);

				for (auto const& [name, expr] : rule.Logic)
				{
					Evaluator.Vars[i][name] = Evaluator.Eval(i, expr);
				}

				for (auto const& phase : rule.Phases)
				{
					if (Evaluator.Eval(i, phase.Condition))
					{
						for (auto const& [fname, fexpr] : phase.Formulas)
						{
							Evaluator.Vars[i][fname] = Evaluator.Eval(i, fexpr);
						}
					}
				}

				std::cout << "DEBUG: " << binding.BoneName << " Progress 1: " << Evaluator.Vars[i].at("t_one") << std::endl;
				std::cout << "DEBUG: " << binding.BoneName << " X: " << Evaluator.Vars[i].at("offset_x") << std::endl;
				std::cout << "DEBUG: " << binding.BoneName << " Z: " << Evaluator.Vars[i].at("offset_z") << std::endl;

				auto outputBlock = RigDesc.FindOutputBlock(RigDesc.Bindings[i].BoneName);
				IdealTargets[0][outputBlock.Offset + 0] = static_cast<float>(Evaluator.Vars[i].at("offset_x") );
				IdealTargets[0][outputBlock.Offset + 1] = static_cast<float>(Evaluator.Vars[i].at("offset_y"));
				IdealTargets[0][outputBlock.Offset + 2] = static_cast<float>(Evaluator.Vars[i].at("offset_z") );
			}
		}

		IdealTarg = IdealTargets;

		auto PosLoss = torch::mse_loss(Pred, IdealTargets);
		auto Zero = torch::zeros({1}, torch::kFloat).to(Input.device());
		auto TotalLoss = (PosLoss * 5.0f);
		return IKLossResult(TotalLoss, PosLoss, IdealTargets, TotalLoss);
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
