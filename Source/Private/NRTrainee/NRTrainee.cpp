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
		int32_t InCount = RigDesc.GetRequiredInputSize() + RigDesc.GetRequiredTargetsSize();
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
	AnalyticResult NRTrainee<T>::SolveAnalyticIK(
		const torch::Tensor& HipPos,
		const torch::Tensor& TargetPos,
		float L1,
		float L2,
		const torch::Tensor& Axis,
		const torch::Tensor& Axis2,
		const torch::Tensor& PoleVec)
	{
		auto ToTarget = TargetPos - HipPos;
		auto Dist = torch::norm(ToTarget, 2, 1, true);

		// Clamp para evitar triângulos impossíveis
		auto D = torch::clamp(Dist, std::abs(L1 - L2) + 1e-2f, (L1 + L2) - 1e-2f);
		auto TargetDir = ToTarget / (D + 1e-6f);

		// 2. Lei dos Cossenos (Mantendo [B, 1])
		auto CosB = (L1 * L1 + L2 * L2 - D.pow(2)) / (2.0f * L1 * L2);
		auto AngleB = torch::acos(torch::clamp(CosB, -1.0f, 1.0f));

		auto CosA = (L1 * L1 + D.pow(2) - L2 * L2) / (2.0f * L1 * D);
		auto AngleA = torch::acos(torch::clamp(CosA, -1.0f, 1.0f));

		// 3. Base Ortonormal
		auto RightVec = torch::linalg_cross(TargetDir, Axis2);
		auto Mag = torch::norm(RightVec, 2, 1, true);

		// Fallback para evitar divisão por zero se o alvo estiver na linha do Pole
		auto UpVector = torch::tensor({0.0f, 1.0f, 0.0f}, TargetDir.options()).expand_as(TargetDir);
		auto FallbackRight = torch::linalg_cross(TargetDir, UpVector);

		auto BendAxis = torch::where(Mag > 1e-4f,
		                             RightVec / (Mag + 1e-6f),
		                             FallbackRight / (torch::norm(FallbackRight, 2, 1, true) + 1e-6f));

		// 4. Coxa (Thigh) - Uso de B_Axis para garantir consistência
		auto IsNegativePrimary = Axis2.select(1, 0) < 0; // Resultado: [Batch]

		// Invertemos o TargetDir se o eixo principal for negativo (ex: perna esquerda)
		auto AdjustedTargetDir = torch::where(IsNegativePrimary.unsqueeze(1), -TargetDir, TargetDir);
		auto LookAtQuat = QuatLookAt(Axis2, AdjustedTargetDir);

		// 5. Joelho (Calf) - Tratando Secondary Axis
		auto IsNegativeSecondary = Axis2.select(1, 1) < 0; // [Batch]

		// CORREÇÃO: Usamos a máscara expandida em vez de squeeze/unsqueeze instáveis
		auto FinalAngleA = torch::where(IsNegativeSecondary.unsqueeze(1), -AngleA, AngleA);
		auto FinalAngleB = torch::where(IsNegativeSecondary.unsqueeze(1), -AngleB, AngleB);

		auto RotatorA = QuatFromAxisAngle(BendAxis, FinalAngleA);
		auto ThighFinal = NormalizeQuats(QuatMultiply(RotatorA, LookAtQuat));
		auto CalfFinal = QuatFromAxisAngle(Axis2, FinalAngleB);

		return {ThighFinal, CalfFinal};
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& Input, const torch::Tensor& Pred)
	{
		auto L1_R = RigDesc.GetInputBoneValue(Input, "bone_l1_r");
		auto L2_R = RigDesc.GetInputBoneValue(Input, "bone_l2_r");
		auto L1_L = RigDesc.GetInputBoneValue(Input, "bone_l1_l");
		auto L2_L = RigDesc.GetInputBoneValue(Input, "bone_l2_l");

		auto IdealFootR = RigDesc.GetInputBoneValue(Input, "foot_r_prevue", true);
		auto IdealFootL = RigDesc.GetInputBoneValue(Input, "foot_l_prevue", true);

		auto PredRQ1 = RigDesc.GetOutputBoneValue(Pred, "thigh_r_quat_out");
		auto PredRQ2 = RigDesc.GetOutputBoneValue(Pred, "calf_r_quat_out");

		auto PredLQ1 = RigDesc.GetOutputBoneValue(Pred, "thigh_l_quat_out");
		auto PredLQ2 = RigDesc.GetOutputBoneValue(Pred, "calf_l_quat_out");

		auto PredRQ3 = RigDesc.GetOutputBoneValue(Pred, "foot_r_quat_out");
		auto PredLQ3 = RigDesc.GetOutputBoneValue(Pred, "foot_l_quat_out");

		std::string Message = "";
		auto PredFootR = RigDesc.GetOutputBoneValue(Pred, "foot_r");
		auto PredFootL = RigDesc.GetOutputBoneValue(Pred, "foot_l");
		auto PredPelvis= RigDesc.GetOutputBoneValue(Pred, "pelvis_pos");

		std::cout << "===========FootR==========" << std::endl;
		Message = "FootR";
		RigDesc.Debug(Message, IdealFootR);
		Message = "PRED: FootR";
		RigDesc.Debug(Message, PredFootR);

		Message = "PRED: PredRQ1";
		RigDesc.Debug(Message, PredRQ1);

		Message = "PRED: PredRQ2";
		RigDesc.Debug(Message, PredRQ2);

		std::cout << "===========FootL==========" << std::endl;
		Message = "FootL";
		RigDesc.Debug(Message, IdealFootL);
		Message = "PRED: FootL";
		RigDesc.Debug(Message, PredFootL);

		Message = "PRED: PredLQ1";
		RigDesc.Debug(Message, PredLQ1);

		Message = "PRED: PredLQ2";
		RigDesc.Debug(Message, PredLQ2);
		std::cout << "==========================" << std::endl;

		auto [TopPosR_P, OffsetR_x] = ForwardKinematicsChain(PredRQ1, PredRQ2, L1_R, L2_R, PredPelvis);
		auto [TopPosL_P, OffsetL_x] = ForwardKinematicsChain(PredLQ1, PredLQ2, L1_L, L2_L, PredPelvis);


		auto PosLossR = torch::mse_loss(PredFootR, IdealFootR);
		auto PosLossL = torch::mse_loss(PredFootL, IdealFootL);

		auto TotalPosLoss = PosLossR + PosLossL;

		auto UnitLoss = [](at::Tensor q) {
			auto n = torch::norm(q, 2, 1); // Shape [Batch]
			return torch::mse_loss(n, torch::ones_like(n)); // Compara [Batch] com [Batch]
		};

		// Quat unit constraint
		auto q1_r_norm = NormalizeQuats(PredRQ1);
		auto q2_r_norm = NormalizeQuats(PredRQ2);
		auto q3_r_norm = NormalizeQuats(PredRQ3);
		auto q1RUnit  = UnitLoss(PredRQ1);
		auto q2RUnit  = UnitLoss(PredRQ2);
		auto q3RUnit  = UnitLoss(PredRQ3);

		auto q1_l_norm = NormalizeQuats(PredLQ1);
		auto q2_l_norm = NormalizeQuats(PredLQ2);
		auto q3_l_norm = NormalizeQuats(PredLQ3);
		auto q1LUnit  = UnitLoss(PredLQ1);
		auto q2LUnit  = UnitLoss(PredLQ2);
		auto q3LUnit  = UnitLoss(PredLQ3);

		auto qUnit = (q1RUnit + q2RUnit + q3RUnit + q1LUnit + q2LUnit + q3LUnit);
		auto TotalLoss = (TotalPosLoss * 5.0) + (qUnit * 0.01f);

		auto Zero = torch::tensor({0.0f}, torch::kFloat);
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
