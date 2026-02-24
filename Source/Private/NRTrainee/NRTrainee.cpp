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

		auto [TotalLoss, PosLoss, RotationLoss, RegLoss] = ComputeIKLoss(InputTensor, Prediction, L1_R, L2_R, L1_L, L2_L);
		TotalLoss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);
		Optimizer->step();

		return TotalLoss.template item<float>();
	}

	template<FloatingPoint T>
	FKResult NRTrainee<T>::ForwardKinematicsChain(
		const torch::Tensor& ThighOffset,
		float L1,
		float L2,
		const torch::Tensor& ThighQuat,
		const torch::Tensor& CalfQuat,
		const torch::Tensor& BoneAxis)
	{
		const auto& HipPosPS = ThighOffset;

		auto ThighQuatNorm = NormalizeQuats(ThighQuat);
		auto ThighDir = QuatRotateVector(ThighQuatNorm, BoneAxis);

		auto KneePosPS = HipPosPS + ThighDir * L1;

		// Calculate the Cumulative Rotation of the Shin (Thigh * Shin)
		// CalfQuantity is local, multiply to get the parent space (Thigh)
		auto CalfQuatPS = NormalizeQuats(QuatMultiply(ThighQuatNorm, CalfQuat));

		// Final Foot Position
		auto CalfDir = QuatRotateVector(CalfQuatPS, BoneAxis);
		auto FootPosPS = KneePosPS + CalfDir * L2;

		return {FootPosPS, KneePosPS};
	}

	template<FloatingPoint T>
	AnalyticResult NRTrainee<T>::SolveAnalyticIK(const torch::Tensor& HipPos, const torch::Tensor& TargetPos, float L1, float L2, const torch::Tensor& BoneAxis, const torch::Tensor& PoleVec)
	{
		auto ToTarget = TargetPos - HipPos;
		auto Dist = torch::norm(ToTarget, 2, 1, true);

		auto MaxDist = L1 + L2 - 0.01f;
		auto MinDist = std::abs(L1 - L2) + 0.01f;
		auto D = torch::clamp(Dist, MinDist, MaxDist);

		auto CosB = (L1 * L1 + L2 * L2 - D.pow(2)) / (2.0f * L1 * L2);
		auto AngleB = torch::acos(torch::clamp(CosB, -1.0f, 1.0f));

		auto CosA = (L1 * L1 + D.pow(2) - L2 * L2) / (2.0f * L1 * D);
		auto AngleA = torch::acos(torch::clamp(CosA, -1.0f, 1.0f));

		// Orientation Construction (Leg Triangle)
		auto TargetDir = ToTarget / D;
		auto RightVec = torch::linalg_cross(TargetDir, PoleVec);
		auto UpVec = torch::linalg_cross(RightVec, TargetDir);
		UpVec = UpVec / torch::norm(UpVec, 2, 1, true);

		// Bending axis (Normal to the triangle)
		auto BendAxis = torch::linalg_cross(TargetDir, UpVec);
		BendAxis = BendAxis / torch::norm(BendAxis, 2, 1, true);

		// Thigh: Rotation to look at the target + Rotation of angle A
		auto LookAtQuat = QuatLookAt(BoneAxis, TargetDir);
		auto RotatorA = QuatFromAxisAngle(BendAxis, AngleA);
		auto ThighFinal = QuatMultiply(LookAtQuat, RotatorA);

		// Calf: Only bend at angle B on the bend axis.
		auto CalfFinal = QuatFromAxisAngle(BendAxis, AngleB);

		return { ThighFinal, CalfFinal };
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeIKLoss(const torch::Tensor& Input, const torch::Tensor& PredQuats, float L1_R, float L2_R, float L1_L, float L2_L)
	{
		auto HipPosR = Input.slice(1, 0, 3);
		auto HipPosL = Input.slice(1, 3, 6);
		auto HasHitR = Input.slice(1, 13, 16);
		auto HasHitL = Input.slice(1, 16, 19);
		auto PoleVec = PredQuats.slice(1, 21, 24);

		auto AxisR = Input.slice(1, 24, 27);
		auto AxisL = Input.slice(1, 27, 30);

		auto FootTargetR = Input.slice(1, 34, 37);
		auto FootTargetL = Input.slice(1, 37, 40);

		// AI prediction of the right leg
		auto PredThighR = PredQuats.slice(1, 0, 4);
		auto PredCalfR  = PredQuats.slice(1, 4, 8);

		// AI prediction of the left leg
		auto PredThighL  = PredQuats.slice(1, 8, 12);
		auto PredCalfL  = PredQuats.slice(1, 12, 16);

		// Forward Kinematics (FK)
		FKResult FK_R = ForwardKinematicsChain(HipPosR, L1_R, L2_R, PredThighR, PredCalfR, AxisR);
		FKResult FK_L = ForwardKinematicsChain(HipPosL, L1_L, L2_L, PredThighL, PredCalfL, AxisL);

		auto DistSqR = ((FK_R.FootPos - FootTargetR) * 0.01f).pow(2).sum(1, true);
		auto DistSqL = ((FK_L.FootPos - FootTargetL) * 0.01f).pow(2).sum(1, true);

		AnalyticResult AnalyticR = SolveAnalyticIK(HipPosR, FootTargetR, L1_R, L2_R, AxisR, PoleVec);
		AnalyticResult AnalyticL = SolveAnalyticIK(HipPosL, FootTargetL, L1_L, L2_L, AxisL, PoleVec);

		// Quat loss
		auto QuatRegLoss = torch::tensor(0.0f, Input.options());
		std::vector<torch::Tensor> q_list = {PredThighR, PredCalfR, PredThighL, PredCalfL};
		for (auto& q : q_list)
		{
			QuatRegLoss += (q.norm(2, 1) - 1.0f).pow(2).mean();
		}

		// Position loss
		auto PosLoss = (DistSqR.mean() + DistSqL.mean()) * 0.5f;

		// Analytic Rotation Loss
		auto RotationLossR = (PredThighR - AnalyticR.ThighQuat).pow(2).mean() + (PredCalfR - AnalyticR.CalfQuat).pow(2).mean();
		auto RotationLossL = (PredThighL - AnalyticL.ThighQuat).pow(2).mean() + (PredCalfL - AnalyticL.CalfQuat).pow(2).mean();
		auto RotationLoss = (RotationLossR + RotationLossL) * 0.5f;

		float lambda_rot = 2.0f;
		float lambda_pos = 1.0f;
		float lambda_reg = 0.1f;

		auto TotalLoss = (PosLoss * lambda_pos) + (RotationLoss * lambda_rot) + (QuatRegLoss * lambda_reg);
		return {TotalLoss, PosLoss, RotationLoss, QuatRegLoss};
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
