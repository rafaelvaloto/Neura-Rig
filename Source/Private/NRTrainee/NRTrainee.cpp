// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

#include "NRCore/NRParse.h"
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
	FKResult NRTrainee<T>::ForwardKinematicsChain(
		const torch::Tensor& Q1,
		const torch::Tensor& Q2,
		const torch::Tensor& L1,
		const torch::Tensor& L2,
		const torch::Tensor& Axis)
	{
		auto rad1 = getAngleFromQuat(Q1);
		auto rad2 = getAngleFromQuat(Q2);

		auto h_coxa = L1 * torch::cos(rad1);
		auto w_coxa = L1 * torch::sin(rad1);

		auto h_canela = L2 * torch::cos(rad2);
		auto w_canela = L2 * torch::sin(rad2);

		// 4. Posição Final da Pelve
		auto TopPos_P = h_canela + h_coxa;
		auto Offset_x = w_canela + w_coxa;

		return {TopPos_P, Offset_x};
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& Input, const torch::Tensor& Pred)
	{
		auto L1_cm1 = RigDesc.GetInputBoneValue(Input, "bone_l1_cm1_one");
		auto L2_cm1 = RigDesc.GetInputBoneValue(Input, "bone_l2_cm1_two");
		auto L1_cm2 = RigDesc.GetInputBoneValue(Input, "bone_l1_cm2_one");
		auto L2_cm2 = RigDesc.GetInputBoneValue(Input, "bone_l2_cm2_two");

		auto IdealFootR = torch::tensor({0.0f, 0.0f, 0.0f}, torch::kFloat);
		auto IdealFootL = torch::tensor({0.0f, 0.0f, 0.0f}, torch::kFloat);

		std::string Message = "";
		auto PredFootR = RigDesc.GetOutputBoneValue(Pred, "bone_vec1_one");
		auto PredFootL = RigDesc.GetOutputBoneValue(Pred, "bone_vec2_one");

		std::cout << "===========FootR==========" << std::endl;
		Message = "FootR";
		RigDesc.Debug(Message, IdealFootR);
		Message = "PRED: FootR";
		RigDesc.Debug(Message, PredFootR);

		std::cout << "===========FootL==========" << std::endl;
		Message = "FootL";
		RigDesc.Debug(Message, IdealFootL);
		Message = "PRED: FootL";
		RigDesc.Debug(Message, PredFootL);
		std::cout << "==========================" << std::endl;


		auto PosLossR = torch::mse_loss(PredFootR, IdealFootR);
		auto PosLossL = torch::mse_loss(PredFootL, IdealFootL);

		auto TotalPosLoss = PosLossR + PosLossL;

		auto UnitLoss = [](at::Tensor q) {
			auto n = torch::norm(q, 2, 1); // Shape [Batch]
			return torch::mse_loss(n, torch::ones_like(n)); // Compara [Batch] com [Batch]
		};

		// Quat unit constraint
		// auto q1_r_norm = NormalizeQuats(PredRQ1);
		// auto q2_r_norm = NormalizeQuats(PredRQ2);
		// auto q3_r_norm = NormalizeQuats(PredRQ3);
		// auto q1RUnit  = UnitLoss(PredRQ1);
		// auto q2RUnit  = UnitLoss(PredRQ2);
		// auto q3RUnit  = UnitLoss(PredRQ3);
		//
		// auto q1_l_norm = NormalizeQuats(PredLQ1);
		// auto q2_l_norm = NormalizeQuats(PredLQ2);
		// auto q3_l_norm = NormalizeQuats(PredLQ3);
		// auto q1LUnit  = UnitLoss(PredLQ1);
		// auto q2LUnit  = UnitLoss(PredLQ2);
		// auto q3LUnit  = UnitLoss(PredLQ3);

		auto Zero = torch::tensor({0.0f}, torch::kFloat);
		auto TotalLoss = (TotalPosLoss * 5.0) + (Zero * 0.01f);
		return IKLossResult(TotalLoss, TotalPosLoss, Zero, TotalLoss);
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
