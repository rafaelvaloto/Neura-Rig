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
	float NRTrainee<T>::TrainStep(const std::vector<float>& InputFloats)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize() + RigDesc.GetRequiredTargetsSize();
		int32_t BatchSize = static_cast<int32_t>(InputFloats.size()) / InCount;

		auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
		auto InputTensor = torch::from_blob((void*)InputFloats.data(), {BatchSize, InCount}, options).clone();

		Optimizer->zero_grad();
		auto Prediction = TargetModel->Forward(InputTensor);
		auto [TotalLoss, PosLoss, RotationLoss, RegLoss] = ComputeIKLoss(InputTensor, Prediction);
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
		const torch::Tensor& Axis)
	{
		// 2. Thigh
		auto ThighQuatNorm = NormalizeQuats(ThighQuat);
		auto ThighDir = QuatRotateVector(ThighQuatNorm, Axis);
		auto KneePosPS = ThighOffset + ThighDir * L1 ;

		// 3. Calf
		// Rotação acumulada: Thigh * Calf
		auto CalfQuatPS = NormalizeQuats(QuatMultiply(ThighQuatNorm, CalfQuat));
		auto CalfDir = QuatRotateVector(CalfQuatPS, Axis);
		auto FootPosPS = KneePosPS + CalfDir * L2 ;

		return {FootPosPS, KneePosPS};
	}

	template<FloatingPoint T>
	AnalyticResult NRTrainee<T>::SolveAnalyticIK(
		const torch::Tensor& HipPos,
		const torch::Tensor& TargetPos,
		float L1, float L2,
		const torch::Tensor& Axis,
		const torch::Tensor& Axis2,
		const torch::Tensor& PoleVec)
	{
		auto B_Axis = Axis.view({-1, 3});
		auto B_Axis2 = Axis2.view({-1, 3});
		auto B_Pole = PoleVec.view({-1, 3});

		auto ToTarget = TargetPos - HipPos;
		auto Dist = torch::norm(ToTarget, 2, 1, true);
		auto D = torch::clamp(Dist, std::abs(L1 - L2) + 0.01f, L1 + L2 - 0.01f);
		auto TargetDir = ToTarget / (D + 1e-6f);

		// 2. Lei dos Cossenos
		auto CosB = (L1 * L1 + L2 * L2 - D.pow(2)) / (2.0f * L1 * L2);
		auto AngleB = torch::acos(torch::clamp(CosB, -1.0f, 1.0f));
		auto CosA = (L1 * L1 + D.pow(2) - L2 * L2) / (2.0f * L1 * D);
		auto AngleA = torch::acos(torch::clamp(CosA, -1.0f, 1.0f));

		// 3. Base Ortonormal
		auto RightVec = torch::linalg_cross(TargetDir, B_Pole);
		auto Mag = torch::norm(RightVec, 2, 1, true);
		auto FallbackRight = torch::linalg_cross(TargetDir, torch::tensor({0.0f, 1.0f, 0.0f}, TargetDir.options()).expand_as(TargetDir));
		auto BendAxis = torch::where(Mag > 1e-4f, RightVec / (Mag + 1e-6f), FallbackRight / (torch::norm(FallbackRight, 2, 1, true) + 1e-6f));

		// 4. Coxa (Thigh) - Tratando Primary Axis (X=1 ou X=-1)
		auto IsNegativePrimary = (Axis.select(1, 0) < 0).unsqueeze(1);

		// Se for negativo, invertemos o TargetDir para que o LookAt gire o osso 180 graus
		auto AdjustedTargetDir = torch::where(IsNegativePrimary, -TargetDir, TargetDir);
		auto LookAtQuat = QuatLookAt(Axis, AdjustedTargetDir);

		// 5. Joelho (Calf) - Tratando Secondary Axis (Y=1 ou Y=-1)
		auto IsNegativeSecondary = B_Axis2.select(1, 1) < 0;
		auto FinalAngleA = torch::where(IsNegativeSecondary, -AngleA.squeeze(), AngleA.squeeze()).unsqueeze(1);
		auto FinalAngleB = torch::where(IsNegativeSecondary, -AngleB.squeeze(), AngleB.squeeze()).unsqueeze(1);

		auto RotatorA = QuatFromAxisAngle(BendAxis, FinalAngleA);
		auto ThighFinal = NormalizeQuats(QuatMultiply(RotatorA, LookAtQuat));
		auto CalfFinal = QuatFromAxisAngle(B_Axis2, FinalAngleB);

		return { ThighFinal, CalfFinal };
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeIKLoss(const torch::Tensor& Input, const torch::Tensor& PredQuats)
	{
		auto HipPosR = Input.slice(1, 0, 3);
		auto HipPosL = Input.slice(1, 3, 6);
		auto HasHitR = Input.slice(1, 13, 14);
		auto HasHitL = Input.slice(1, 17, 18);
		auto PoleVecR = Input.slice(1, 21, 24);
		auto PoleVecL = Input.slice(1, 24, 27);

		auto AxisR = Input.slice(1, 27, 30);
		auto AxisL = Input.slice(1, 30, 33);
		auto Axis2R = Input.slice(1, 33, 36);
		auto Axis2L = Input.slice(1, 36, 39);

		auto FootTargetR = Input.slice(1, 43, 46);
		auto FootTargetL = Input.slice(1, 46, 49);
		auto PelvisPos  = Input.slice(1, 49, 52);

		auto L1_R = Input.slice(1, 6, 7).item<float>();
		auto L2_R = Input.slice(1, 7, 8).item<float>();
		auto L1_L = Input.slice(1, 8, 9).item<float>();
		auto L2_L = Input.slice(1, 9, 10).item<float>();

		// AI prediction of the right leg
		auto PredThighR = PredQuats.slice(1, 0, 4);
		auto PredCalfR = PredQuats.slice(1, 4, 8);

		// AI prediction of the left leg
		auto PredThighL = PredQuats.slice(1, 12, 16);
		auto PredCalfL = PredQuats.slice(1, 16, 20);

		// AI prediction PredQuats
		auto PredPelvisQuat = PredQuats.slice(1, 24, 28);
		auto PelvisQuatNorm = NormalizeQuats(PredPelvisQuat);

		auto RotatedOffset = QuatRotateVector(PelvisQuatNorm, HipPosR);
		auto ActualHipPosR = PelvisPos + RotatedOffset;
		auto RotatedOffsetL = QuatRotateVector(PelvisQuatNorm, HipPosL);
		auto ActualHipPosL = PelvisPos + RotatedOffsetL;

		// std::cout << "ActualHipPosR: " << ActualHipPosR << std::endl;
		// std::cout << "ActualHipPosL: " << ActualHipPosL << std::endl;

		// Forward Kinematics (FK)
		FKResult FK_R = ForwardKinematicsChain(ActualHipPosR, L1_R, L2_R, PredThighR, PredCalfR, AxisR);
		FKResult FK_L = ForwardKinematicsChain(ActualHipPosL, L1_L, L2_L, PredThighL, PredCalfL, AxisL);

		auto DistSqR = ((FK_R.FootPos - FootTargetR) * 0.01f).pow(2).sum(1, true);
		auto DistSqL = ((FK_L.FootPos - FootTargetL) * 0.01f).pow(2).sum(1, true);

		AnalyticResult AnalyticR = SolveAnalyticIK(ActualHipPosR, FootTargetR, L1_R, L2_R, AxisR, Axis2R, PoleVecR);
		AnalyticResult AnalyticL = SolveAnalyticIK(ActualHipPosL, FootTargetL, L1_L, L2_L, AxisL, Axis2L, PoleVecL);

		// Position loss
		auto PosLoss = (DistSqR.mean() + DistSqL.mean()) * 0.5f;

		// Analytic Rotation Loss
		auto RotationLossR = (PredThighR - AnalyticR.ThighQuat).pow(2).mean() + (PredCalfR - AnalyticR.CalfQuat).pow(2).mean();
		auto RotationLossL = (PredThighL - AnalyticL.ThighQuat).pow(2).mean() + (PredCalfL - AnalyticL.CalfQuat).pow(2).mean();
		auto RotationLoss = (RotationLossR + RotationLossL) * 0.5f;

		// Quat loss
		auto QuatRegLoss = torch::tensor(0.0f, Input.options());
		std::vector<torch::Tensor> q_list = {PredPelvisQuat, PredThighR, PredCalfR, PredThighL, PredCalfL};
		for (auto& q : q_list)
		{
			QuatRegLoss += (q.norm(2, 1) - 1.0f).pow(2).mean();
		}

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
