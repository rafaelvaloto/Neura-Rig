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
		if (BatchSize == 0) return 0.0f;

		auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
		auto InputTensor = torch::from_blob((void*)InputFloats.data(), {BatchSize, InCount}, options).clone();

		Optimizer->zero_grad();

		// Modelo prediz 24 floats (6 quats)
		auto Prediction = TargetModel->Forward(InputTensor);

		auto [TotalLoss, PosLoss, RegLoss] = ComputeIKLoss(InputTensor, Prediction, L1_R, L2_R, L1_L, L2_L);

		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);
		Optimizer->step();

		return TotalLoss.item<float>();
	}

	template<FloatingPoint T>
	FKResult NRTrainee<T>::ForwardKinematicsChain(const torch::Tensor& PelvisPos, const torch::Tensor& PelvisQuat, const torch::Tensor& ThighOffset, float L1, float L2, const torch::Tensor& ThighQuat, const torch::Tensor& CalfQuat, const torch::Tensor& BoneAxis)
	{
		// 1) Posição do Hip em WS
		auto HipPosWS = PelvisPos + QuatRotateVector(PelvisQuat, ThighOffset);

		// 2) Thigh quat em WS = pelvis_quat * thigh_quat_ls (predito)
		auto ThighQuatWS = NormalizeQuats(QuatMultiply(PelvisQuat, ThighQuat));

		// 3) Knee = Hip + rotate(ThighQuatWS, ForwardVector) * L1
		// Assumindo que o bone aponta para X no local space (ajustar conforme o rig)
		auto ThighDir = QuatRotateVector(ThighQuatWS, BoneAxis);
		auto KneePosWS = HipPosWS + ThighDir * L1;

		// 4) Calf quat em WS = thigh_quat_ws * calf_quat_ls (predito)
		auto CalfQuatWS = NormalizeQuats(QuatMultiply(ThighQuatWS, CalfQuat));

		// 5) Foot = Knee + rotate(CalfQuatWS, ForwardVector) * L2
		auto CalfDir = QuatRotateVector(CalfQuatWS, BoneAxis);
		auto FootPosWS = KneePosWS + CalfDir * L2;

		return {FootPosWS, KneePosWS};
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeIKLoss(const torch::Tensor& Input, const torch::Tensor& PredQuats, float L1_R, float L2_R, float L1_L, float L2_L)
	{
		int B = Input.size(0);
		auto PelvisP = Input.slice(1, 0, 3);  // (B, 3)
		auto PelvisQuat = Input.slice(1, 3, 7);  // (B, 4)

		// Hip positions (thigh_r_pos offset 7, thigh_l_pos offset 28)
		auto HipPosR = Input.slice(1, 7, 10);  // (B, 3)
		auto HipPosL = Input.slice(1, 28, 31); // (B, 3)

		// Foot targets (offset 57..62)
		auto FootTargetR = Input.slice(1, 57, 60); // (B, 3)
		auto FootTargetL = Input.slice(1, 60, 63); // (B, 3)

		// Has hit flags (offset 52, 56)
		auto HasHitR = Input.slice(1, 52, 53); // (B, 1)
		auto HasHitL = Input.slice(1, 56, 57); // (B, 1)

		// // ── Extrair quats preditos (B, 24) → 6 quats ──
		//
		auto ThighR_Q = NormalizeQuats(PredQuats.slice(1, 0, 4));
		auto CalfR_Q = NormalizeQuats(PredQuats.slice(1, 4, 8));
		auto FootR_Q = NormalizeQuats(PredQuats.slice(1, 8, 12));
		auto ThighL_Q = NormalizeQuats(PredQuats.slice(1, 12, 16));
		auto CalfL_Q = NormalizeQuats(PredQuats.slice(1, 16, 20));
		auto FootL_Q = NormalizeQuats(PredQuats.slice(1, 20, 24));

		// ── Forward Kinematics ──

		auto BoneAxisR = torch::tensor({1.0f, 0.0f, 0.0f}).unsqueeze(0).expand({B, 3});
		auto BoneAxisL = torch::tensor({-1.0f, 0.0f, 0.f}).unsqueeze(0).expand({B, 3});

		FKResult FK_R = ForwardKinematicsChain(PelvisP, PelvisQuat, HipPosR, L1_R, L2_R, ThighR_Q, CalfR_Q, BoneAxisR);
		FKResult FK_L = ForwardKinematicsChain(PelvisP, PelvisQuat, HipPosL, L1_L, L2_L, ThighL_Q, CalfL_Q, BoneAxisL);

		// Converter targets de Pelvis Space → World Space
		auto FootTargetR_WS = PelvisP + QuatRotateVector(PelvisQuat, FootTargetR);
		auto FootTargetL_WS = PelvisP + QuatRotateVector(PelvisQuat, FootTargetL);

		// Loss — agora ambos estão em WS
		auto PosErrorR = (FK_R.FootPos - FootTargetR_WS).pow(2).sum(1, true);
		auto PosErrorL = (FK_L.FootPos - FootTargetL_WS).pow(2).sum(1, true);

		auto MaskedPosErrorR = PosErrorR * HasHitR;
		auto MaskedPosErrorL = PosErrorL * HasHitL;
		
		auto PosLoss = (MaskedPosErrorR.sum() + MaskedPosErrorL.sum()) / (HasHitR.sum() + HasHitL.sum() + 0.01f);

		// ── Regularização: quats devem ser unitários ──

		auto QuatRegLoss = torch::tensor(0.0f, Input.options());
		for (auto& q : {ThighR_Q, CalfR_Q, FootR_Q, ThighL_Q, CalfL_Q, FootL_Q})
		{
			auto norm = q.norm(2, 1);
			QuatRegLoss = QuatRegLoss + (norm - 1.0f).pow(2).mean();
		}

		// ── Loss total ──

		float lambda_reg = 0.01f;
		auto TotalLoss = PosLoss + lambda_reg * QuatRegLoss;

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
