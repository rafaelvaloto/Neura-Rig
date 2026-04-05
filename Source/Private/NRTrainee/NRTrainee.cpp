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
		auto [TotalLoss] = ComputeRLReward(InputTensor, Prediction);

		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(true), 1.0);

		Optimizer->step();
		return TotalLoss.template item<float>();
	}

	int frameCounter = 0;

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& InputTensor, const torch::Tensor& Pred)
	{
		constexpr float wPos = 1.0f;
		constexpr float wPelvis = 1.0f;
		constexpr float wFK = 0.1f;
		constexpr float wTemporal = 0.35f;
		constexpr float wAccel = 0.20f;
		constexpr float wSmoothOut = 0.15f;

		const auto options = Pred.options();
		auto T_ideal = torch::zeros_like(Pred);

		const auto B_size = RigDesc.Bindings.size();

		for (size_t i = 0; i < B_size; ++i)
		{
			auto& varMap = Evaluator.Parsers[i].GetVar();
			auto& Bindings = RigDesc.Bindings[i];

			for (const auto& F_rule : Bindings.Rules)
			{
				if (F_rule.Name.empty())
				{
					std::cerr << "[NRTrainee] Rule not found: " << Bindings.RuleName << std::endl;
					continue;
				}

				Evaluator.SetTensorInputs(i, F_rule, RigDesc, InputTensor);

				for (const auto& [Name, Expr] : F_rule.Logic)
				{
					auto it = varMap.find(Name);
					if (it == varMap.end())
					{
						std::cerr << "[NRTrainee] Logic variable not found: " << Name << std::endl;
						continue;
					}
					*(it->second) = Evaluator.Eval(i, Expr);
				}

				for (const auto& [Id, Condition, Formulas] : F_rule.Phases)
				{
					auto condName = Id + "_condition";
					auto it = varMap.find(condName);
					if (it == varMap.end())
					{
						std::cerr << "[NRTrainee] Phase condition variable not found: " << condName << std::endl;
						continue;
					}

					*(it->second) = Evaluator.Eval(i, Condition);
					if (*(it->second) == 0)
						continue;

					int B_idx = 0;
					for (const auto& [formulaName, expr] : Formulas)
					{
						auto key = Id + "_" + formulaName;
						if (auto F_it = varMap.find(key); F_it != varMap.end())
						{
							*(F_it->second) = Evaluator.Eval(i, expr);
							T_ideal[0][Bindings.Offset + B_idx] = *(F_it->second);
						}
						++B_idx;
					}

					break;
				}
			}
		}

		std::vector<int64_t> ROffsets;
		std::vector<int64_t> LOffsets;
		ROffsets.reserve(2);
		LOffsets.reserve(2);

		for (size_t i = 0; i < B_size; ++i)
		{
			auto& Bindings = RigDesc.Bindings[i];
			if (Bindings.BoneName == "calf_r" || Bindings.BoneName == "thigh_r")
				ROffsets.push_back(Bindings.Offset + 3);
			else if (Bindings.BoneName == "calf_l" || Bindings.BoneName == "thigh_l")
				LOffsets.push_back(Bindings.Offset + 3);
		}

		// =========================
		// Loss base
		// =========================
		auto P_Pelvis = Pred.index({0, torch::indexing::Slice(48, 54)});
		auto T_Pelvis = T_ideal.index({0, torch::indexing::Slice(48, 54)});
		auto PelvisLoss = torch::mse_loss(P_Pelvis, T_Pelvis);

		auto P1_xyz = Pred.index({0, torch::indexing::Slice(0, 3)});
		auto P2_xyz = Pred.index({0, torch::indexing::Slice(6, 9)});
		auto T1_xyz = T_ideal.index({0, torch::indexing::Slice(0, 3)});
		auto T2_xyz = T_ideal.index({0, torch::indexing::Slice(6, 9)});

		auto Pos1Loss = torch::mse_loss(P1_xyz, T1_xyz);
		auto Pos2Loss = torch::mse_loss(P2_xyz, T2_xyz);

		auto P_FK = ValidateFeetFK(Pred, LOffsets, ROffsets, true);
		auto FKLoss = P_FK.err_loss * wFK;

		auto BaseLoss = (Pos1Loss + Pos2Loss * 2.0f) * wPos + PelvisLoss * wPelvis + FKLoss;

		// =========================
		// Temporal smoothing loss
		// =========================
		auto TemporalLoss = torch::tensor(0.0f, options);
		auto AccelLoss = torch::tensor(0.0f, options);
		auto SmoothOutLoss = torch::tensor(0.0f, options);

		if (PredHistory.defined() && PredHistory.numel() > 0)
		{
			TemporalLoss = torch::mse_loss(Pred, PredHistory);

			if (PredHistory2.defined() && PredHistory2.numel() > 0)
			{
				auto velNow = Pred - PredHistory;
				auto velPrev = PredHistory - PredHistory2;
				AccelLoss = torch::mse_loss(velNow, velPrev);
			}
		}

		// Suavização opcional da saída usada pela animação
		if (SmoothedOutput.defined() && SmoothedOutput.numel() > 0)
		{
			SmoothOutLoss = torch::mse_loss(Pred, SmoothedOutput);
		}

		// =========================
		// Limites / consistência física
		// =========================
		auto ConstraintLoss = torch::tensor(0.0f, options);

		auto clampPenalty = [&](const torch::Tensor& v, float mn, float mx) {
			auto low = torch::clamp(mn - v, 0.0f);
			auto high = torch::clamp(v - mx, 0.0f);
			return low.pow(2).sum() + high.pow(2).sum();
		};

		auto TotalLoss =
			BaseLoss +
			TemporalLoss * wTemporal +
			AccelLoss * wAccel +
			SmoothOutLoss * wSmoothOut;


		// =========================
		// Atualiza histórico
		// =========================
		PredHistory2 = PredHistory.defined() ? PredHistory.detach().clone() : torch::Tensor();
		PredHistory = Pred.detach().clone();

		// Suavização EMA para uso em runtime/animação
		constexpr float emaAlpha = 0.30f;
		if (!SmoothedOutput.defined() || SmoothedOutput.numel() == 0)
		{
			SmoothedOutput = Pred.detach().clone();
		}
		else
		{
			SmoothedOutput = SmoothedOutput * (1.0f - emaAlpha) + Pred.detach() * emaAlpha;
		}

		Predicated = Pred;
		IdealTargets = T_ideal;

		if (frameCounter++ % 30 == 0)
		{
			std::cout << "Frame: " << frameCounter << std::endl;
			std::cout << "BaseLoss: " << BaseLoss.item<float>() << std::endl;
			std::cout << "TemporalLoss: " << TemporalLoss.item<float>() << std::endl;
			std::cout << "AccelLoss: " << AccelLoss.item<float>() << std::endl;
			std::cout << "TotalLoss: " << TotalLoss.item<float>() << std::endl;
		}

		return IKLossResult(TotalLoss);
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
