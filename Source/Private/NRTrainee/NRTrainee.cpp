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
		auto [TotalLoss, PosLoss, RotationLoss, RegLoss] = ComputeRLReward(InputTensor, Prediction);

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
		auto KneePosPS = ThighOffset + ThighDir * L1;

		// 3. Calf
		// Rotação acumulada: Thigh * Calf
		auto CalfQuatPS = NormalizeQuats(QuatMultiply(ThighQuatNorm, CalfQuat));
		auto CalfDir = QuatRotateVector(CalfQuatPS, Axis);
		auto FootPosPS = KneePosPS + CalfDir * L2;

		return {FootPosPS, KneePosPS};
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


	// --- O MAESTRO (COMPUTE RL REWARD) ---
	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeRLReward(const torch::Tensor& Input, const torch::Tensor& Pred)
	{
		// --- 1. FUNÇÕES DE UTILIDADE E CONVERSÃO ---
		auto SafeNormalize = [](torch::Tensor q) {
			return q / (torch::norm(q, 2, -1, true) + 1e-7);
		};

		const float CM_TO_METERS = 0.01f;

		// --- 2. EXTRAÇÃO E CONVERSÃO DE UNIDADES (CM -> SME) ---
		// Multiplicamos os comprimentos dos ossos e posições por 0.01
		auto L1_R = RigDesc.GetInputBoneValue(Input, "bone_l1_r") * CM_TO_METERS;
		auto L2_R = RigDesc.GetInputBoneValue(Input, "bone_l2_r") * CM_TO_METERS;
		auto L1_L = RigDesc.GetInputBoneValue(Input, "bone_l1_l") * CM_TO_METERS;
		auto L2_L = RigDesc.GetInputBoneValue(Input, "bone_l2_l") * CM_TO_METERS;

		// Posições e Targets (CM -> SME)
		auto IdealFootR = RigDesc.GetInputBoneValue(Input, "foot_r_prevue", true) ;
		auto IdealFootL = RigDesc.GetInputBoneValue(Input, "foot_l_prevue", true) ;
		auto HipPosR_Rel = RigDesc.GetInputBoneValue(Input, "thigh_r_pos") ;
		auto HipPosL_Rel = RigDesc.GetInputBoneValue(Input, "thigh_l_pos") ;

		// --- 3. EXTRAÇÃO DAS PREDIÇÕES (OUTPUTS) ---
		auto pPelvisPos = RigDesc.GetOutputBoneValue(Pred, "pelvis_pos");
		auto pPelvisQuat = SafeNormalize(RigDesc.GetOutputBoneValue(Pred, "pelvis_quat_out"));

		// Rotações das pernas
		auto pThighR = SafeNormalize(RigDesc.GetOutputBoneValue(Pred, "thigh_r_quat_out"));
		auto pCalfR = SafeNormalize(RigDesc.GetOutputBoneValue(Pred, "calf_r_quat_out"));
		auto pThighL = SafeNormalize(RigDesc.GetOutputBoneValue(Pred, "thigh_l_quat_out"));
		auto pCalfL = SafeNormalize(RigDesc.GetOutputBoneValue(Pred, "calf_l_quat_out"));

		// --- 4. SIMULAÇÃO DO AMBIENTE (WORLD-ISH SPACE) ---
		auto ActualHipR = pPelvisPos + QuatRotateVector(pPelvisQuat, HipPosR_Rel);
		auto ActualHipL = pPelvisPos + QuatRotateVector(pPelvisQuat, HipPosL_Rel);

		// --- 5. PROTEÇÃO CONTRA OVERREACH (Onde o NaN morre) ---
		auto ClampTarget = [&](torch::Tensor Hip, torch::Tensor Target, float Bone1, float Bone2) {
			auto ToTarget = Target - Hip;
			auto Dist = torch::norm(ToTarget, 2, -1, true);
			float MaxReach = (Bone1 + Bone2) * 0.98f;
			return torch::where(Dist > MaxReach, Hip + (ToTarget / (Dist + 1e-7)) * MaxReach, Target);
		};

		auto ClampedFootR = ClampTarget(ActualHipR, IdealFootR, L1_R.item<float>(), L2_R.item<float>());
		auto ClampedFootL = ClampTarget(ActualHipL, IdealFootL, L1_L.item<float>(), L2_L.item<float>());

		// --- 6. GABARITO ANALÍTICO E FK ---
		// Solver (usando unidades em metros agora)
		auto [AnalyticThighR, AnalyticCalfR] = SolveAnalyticIK(ActualHipR, ClampedFootR, L1_R.item<float>(), L2_R.item<float>(),
		                                                       RigDesc.GetInputBoneValue(Input, "pri_axis_r"), RigDesc.GetInputBoneValue(Input, "sec_axis_r"),
		                                                       SafeNormalize(RigDesc.GetInputBoneValue(Input, "PoleR_PS")));

		auto [AnalyticThighL, AnalyticCalfL] = SolveAnalyticIK(ActualHipL, ClampedFootL, L1_L.item<float>(), L2_L.item<float>(),
		                                                       RigDesc.GetInputBoneValue(Input, "pri_axis_l"), RigDesc.GetInputBoneValue(Input, "sec_axis_l"),
		                                                       SafeNormalize(RigDesc.GetInputBoneValue(Input, "PoleL_PS")));

		// Forward Kinematics para validação
		FKResult FK_R = ForwardKinematicsChain(ActualHipR, L1_R.item<float>(), L2_R.item<float>(), pThighR, pCalfR, RigDesc.GetInputBoneValue(Input, "pri_axis_r"));
		FKResult FK_L = ForwardKinematicsChain(ActualHipL, L1_L.item<float>(), L2_L.item<float>(), pThighL, pCalfL, RigDesc.GetInputBoneValue(Input, "pri_axis_l"));

		// --- 7. CÁLCULO DAS RECOMPENSAS (LOSSES) ---
		auto PosLoss = torch::mse_loss(FK_R.FootPos, IdealFootR) + torch::mse_loss(FK_L.FootPos, IdealFootL);

		auto RotLoss = torch::mse_loss(pThighR, AnalyticThighR) + torch::mse_loss(pCalfR, AnalyticCalfR) +
		               torch::mse_loss(pThighL, AnalyticThighL) + torch::mse_loss(pCalfL, AnalyticCalfL);

		// Regularização: Penaliza se o Quat original estiver muito longe de magnitude 1.0
		auto RegLoss = torch::pow(torch::norm(RigDesc.GetOutputBoneValue(Pred, "thigh_r_quat_out"), 2, -1) - 1.0, 2).mean() +
		               torch::pow(torch::norm(RigDesc.GetOutputBoneValue(Pred, "thigh_l_quat_out"), 2, -1) - 1.0, 2).mean();

		// --- 8. AGREGAÇÃO FINAL ---
		float w_pos = 1.0f;
		float w_rot = 1.0f;
		float w_reg = 0.1f;


		auto TotalLoss = (PosLoss * w_pos) + (RotLoss * w_rot) + (RegLoss * w_reg);

		// Proteção Final contra propagação de NaN
		if (torch::isnan(TotalLoss).item<bool>()) {
			return { torch::tensor(0.0, torch::requires_grad(true)), torch::tensor(0.0), torch::tensor(0.0), torch::tensor(0.0) };
		}

		return { TotalLoss, PosLoss, RotLoss, RegLoss };
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
