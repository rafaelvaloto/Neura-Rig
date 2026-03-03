// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

#include "NRCore/NRRules.h"
#include "NRCore/NRTypes.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T> > TargetModel, NRModelProfile Rig, const double LearningRate)
		: TargetModel(TargetModel)
		  , RigDesc(Rig)
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));
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

		NRRules Evaluator;
		auto IdealTargets = torch::zeros_like(Pred);

		for (int64_t b = 0; b < batchSize; ++b)
		{
			torch::Tensor currentInput = Input[b]; // Pega a linha b (1D: [InCount])

			// Process bindings (foot_r, foot_l)
			for (int i = 0; i < RigDesc.Bindings.size(); ++i)
			{
				auto& binding = RigDesc.Bindings[i];

				// localize rule by name (Ex: Humanoid_Foot_Curve)
				auto rule = RigDesc.FindRule(binding.RuleName);
				Evaluator.Setup(rule, binding, i, RigDesc, currentInput);

				// "Logic" (stride -> cycle -> height)
				for (auto const& [name, expr] : rule.Logic)
				{
					Evaluator.DefineVariable(name);
				}

				for (auto const& [name, expr] : rule.Logic)
				{
					Evaluator.Vars[name] = Evaluator.Eval(expr);
				}

				// (Stance or Swing)
				float finalOffX = 0.0f;
				float finalOffZ = 0.0f;

				for (auto const& phase : rule.Phases)
				{
					// Register formulas before available condition
					for (auto const& [name, expr] : phase.Formulas)
					{
						Evaluator.DefineVariable(name);
					}

					// Condition (ex: cycle < 0.45)
					if (Evaluator.Eval(phase.Condition) > 0.5f)
					{
						for (auto const& [name, expr] : phase.Formulas)
						{
							Evaluator.Vars[name] = Evaluator.Eval(expr);
						}

						finalOffX = static_cast<float>(Evaluator.Vars.at("offset_x"));
						finalOffZ = static_cast<float>(Evaluator.Vars.at("offset_z"));
						break;
					}
				}

				// IdealTargets
				// (Ex: foot_r = offset 0, foot_l = offset 3)
				auto outputBlock = RigDesc.FindOutputBlock(binding.BoneName);

				// [X, Y, Z]
				IdealTargets[b][outputBlock.Offset + 0] = finalOffX;
				IdealTargets[b][outputBlock.Offset + 1] = 0.0f;
				IdealTargets[b][outputBlock.Offset + 2] = finalOffZ;
			}
		}

		// (Pred x IdealTargets)
		auto PosLoss = torch::mse_loss(Pred, IdealTargets);
		auto Zero = torch::zeros({1}, torch::kFloat).to(Input.device());

		auto TotalLoss = (PosLoss * 5.0f);
		return IKLossResult(TotalLoss, PosLoss, Zero, TotalLoss);
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
