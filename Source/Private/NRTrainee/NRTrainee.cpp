// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"
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
	float NRTrainee<T>::TrainStep(const std::vector<float>& InputFloats, float L1_R, float L2_R, float L1_L, float L2_L)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize() + RigDesc.GetRequiredTargetsSize();
		int32_t BatchSize = static_cast<int32_t>(InputFloats.size()) / InCount;

		auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
		auto InputTensor = torch::from_blob((void*)InputFloats.data(), {BatchSize, InCount}, options).clone();

		Optimizer->zero_grad();

		auto Prediction = TargetModel->Forward(InputTensor);

		auto [TotalLoss, PosLoss, RegLoss] = ComputeIKLoss(InputTensor, Prediction, L1_R, L2_R, L1_L, L2_L);
		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);
		Optimizer->step();

		return TotalLoss.template item<float>();
	}

	template<FloatingPoint T>
	FKResult NRTrainee<T>::ForwardKinematicsChain(const torch::Tensor& ThighOffset, float L1, float L2, const torch::Tensor& ThighQuat, const torch::Tensor& CalfQuat, const torch::Tensor& BoneAxis)
	{
		const auto& HipPosPS = ThighOffset;
		auto ThighQuatPS = NormalizeQuats(ThighQuat);

		auto ThighDir = QuatRotateVector(ThighQuatPS, BoneAxis);
		auto KneePosPS = HipPosPS + ThighDir * L1;

		auto CalfQuatPS = NormalizeQuats(QuatMultiply(ThighQuatPS, CalfQuat));

		auto CalfDir = QuatRotateVector(CalfQuatPS, BoneAxis);
		auto FootPosPS = KneePosPS + CalfDir * L2;

		return {FootPosPS, KneePosPS};
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeIKLoss(const torch::Tensor& Input, const torch::Tensor& PredQuats, float L1_R, float L2_R, float L1_L, float L2_L)
	{
		int B = Input.size(0);

		auto HipPosR = Input.slice(1, 7, 10);
		auto HipPosL = Input.slice(1, 28, 31);
		auto HasHitR = Input.slice(1, 52, 53);
		auto HasHitL = Input.slice(1, 56, 57);
		auto FootTargetR = Input.slice(1, 60, 63);
		auto FootTargetL = Input.slice(1, 63, 66);

		auto ThighR_Q = NormalizeQuats(PredQuats.slice(1, 0, 4));
		auto CalfR_Q = NormalizeQuats(PredQuats.slice(1, 4, 8));
		auto FootR_Q = NormalizeQuats(PredQuats.slice(1, 8, 12));

		auto ThighL_Q = NormalizeQuats(PredQuats.slice(1, 12, 16));
		auto CalfL_Q = NormalizeQuats(PredQuats.slice(1, 16, 20));
		auto FootL_Q = NormalizeQuats(PredQuats.slice(1, 20, 24));

		auto BoneAxisR = torch::tensor({1.0f, 0.0f, 0.0f}, Input.options()).unsqueeze(0).expand({B, 3});
		auto BoneAxisL = torch::tensor({-1.0f, 0.0f, 0.0f}, Input.options()).unsqueeze(0).expand({B, 3});

		// Forward Kinematics (FK)
		FKResult FK_R = ForwardKinematicsChain(HipPosR, L1_R, L2_R, ThighR_Q, CalfR_Q, BoneAxisR);
		FKResult FK_L = ForwardKinematicsChain(HipPosL, L1_L, L2_L, ThighL_Q, CalfL_Q, BoneAxisL);

		auto DistSqR = ((FK_R.FootPos - FootTargetR) * 0.01f).pow(2).sum(1, true);
		auto DistSqL = ((FK_L.FootPos - FootTargetL) * 0.01f).pow(2).sum(1, true);

		auto PosLoss = ((DistSqR * HasHitR).sum() + (DistSqL * HasHitL).sum()) / (HasHitR.sum() + HasHitL.sum() + 1e-6f);

		// twist
		auto TwistLoss = (ThighR_Q.slice(1, 0, 1).pow(2) + ThighL_Q.slice(1, 0, 1).pow(2)).mean();

		auto HingeLossR = CalfR_Q.slice(1, 0, 1).pow(2) + CalfR_Q.slice(1, 2, 3).pow(2);
		auto HingeLossL = CalfL_Q.slice(1, 0, 1).pow(2) + CalfL_Q.slice(1, 2, 3).pow(2);
		auto HingeLoss = (HingeLossR + HingeLossL).mean();

		auto FlipLoss = (torch::clamp(-CalfR_Q.slice(1, 1, 2), 0).pow(2) +
		                 torch::clamp(-CalfL_Q.slice(1, 1, 2), 0).pow(2)).mean();

		auto FootStabilityLoss = ((FootR_Q.slice(1, 3, 4) - 1.0f).pow(2) +
		                          (FootL_Q.slice(1, 3, 4) - 1.0f).pow(2)).mean();

		auto QuatRegLoss = torch::tensor(0.0f, Input.options());
		std::vector<torch::Tensor> q_list = {ThighR_Q, CalfR_Q, FootR_Q, ThighL_Q, CalfL_Q, FootL_Q};
		for (auto& q : q_list)
			QuatRegLoss += (q.norm(2, 1) - 1.0f).pow(2).mean();

		float lambda_pos = 1.0f;
		float lambda_reg = 0.1f;
		float lambda_twist = 0.5f;
		float lambda_hinge = 0.5f;
		float lambda_flip = 1.0f;
		float lambda_foot = 0.2f;

		auto TotalLoss = (PosLoss * lambda_pos) +
		                 (QuatRegLoss * lambda_reg) +
		                 (TwistLoss * lambda_twist) +
		                 (HingeLoss * lambda_hinge) +
		                 (FlipLoss * lambda_flip) +
		                 (FootStabilityLoss * lambda_foot);

		return {TotalLoss, PosLoss, QuatRegLoss};
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
