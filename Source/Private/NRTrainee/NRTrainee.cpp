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

		// Modelo prediz 24 floats (6 quats)
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
		// 1) Posição do Hip em WS
		const auto& HipPosPS = ThighOffset;

		// 2) Thigh em PS: como a Pelvis em PS é a identidade,
		// a rotação da coxa em PS é igual à sua rotação local inicial (ou a predita)
		auto ThighQuatPS = NormalizeQuats(ThighQuat);

		// 3) Knee em PS
		auto ThighDir = QuatRotateVector(ThighQuatPS, BoneAxis);
		auto KneePosPS = HipPosPS + ThighDir * L1;

		// 4) Calf em PS = thigh_quat_ps * calf_quat_ls
		auto CalfQuatPS = NormalizeQuats(QuatMultiply(ThighQuatPS, CalfQuat));

		// 5) Foot em PS
		auto CalfDir = QuatRotateVector(CalfQuatPS, BoneAxis);
		auto FootPosPS = KneePosPS + CalfDir * L2;

		return {FootPosPS, KneePosPS};
	}

	template<FloatingPoint T>
	IKLossResult NRTrainee<T>::ComputeIKLoss(const torch::Tensor& Input, const torch::Tensor& PredQuats, float L1_R, float L2_R, float L1_L, float L2_L)
	{
		int B = Input.size(0);

		// 1. Extração de Dados (Offsets corretos conforme o teu schema)
		auto HipPosR = Input.slice(1, 7, 10);
		auto HipPosL = Input.slice(1, 28, 31);
		auto FootTargetR = Input.slice(1, 60, 63);
		auto FootTargetL = Input.slice(1, 63, 66);
		auto HasHitR = Input.slice(1, 52, 53);
		auto HasHitL = Input.slice(1, 56, 57);

		// 2. Extração e Normalização Imediata dos Quaternions Preditos
		// IMPORTANTE: Normalizar aqui garante que a FK use quats válidos
		auto ThighR_Q = NormalizeQuats(PredQuats.slice(1, 0, 4));
		auto CalfR_Q = NormalizeQuats(PredQuats.slice(1, 4, 8));
		auto FootR_Q = NormalizeQuats(PredQuats.slice(1, 8, 12));
		auto ThighL_Q = NormalizeQuats(PredQuats.slice(1, 12, 16));
		auto CalfL_Q = NormalizeQuats(PredQuats.slice(1, 16, 20));
		auto FootL_Q = NormalizeQuats(PredQuats.slice(1, 20, 24));

		// 3. Eixos dos Ossos (Confirma se o teu pé não está a ir para o lado errado!)
		auto BoneAxisR = torch::tensor({1.0f, 0.0f, 0.0f}, Input.options()).unsqueeze(0).expand({B, 3});
		auto BoneAxisL = torch::tensor({-1.0f, 0.0f, 0.0f}, Input.options()).unsqueeze(0).expand({B, 3});

		// 4. Forward Kinematics
		FKResult FK_R = ForwardKinematicsChain(HipPosR, L1_R, L2_R, ThighR_Q, CalfR_Q, BoneAxisR);
		FKResult FK_L = ForwardKinematicsChain(HipPosL, L1_L, L2_L, ThighL_Q, CalfL_Q, BoneAxisL);

		// 5. Cálculo do Erro de Posição (Escalonado)
		// Usamos MSE mas dividimos por 100 para converter a "intensidade" de cm para algo menor
		auto PosErrorR = (FK_R.FootPos - FootTargetR).abs().sum(1, true);
		auto PosErrorL = (FK_L.FootPos - FootTargetL).abs().sum(1, true);

		auto MaskedPosErrorR = PosErrorR * HasHitR;
		auto MaskedPosErrorL = PosErrorL * HasHitL;

		// Média do erro onde houve hit
		auto TotalHit = HasHitR.sum() + HasHitL.sum() + 1e-4f;
		auto PosLoss = (MaskedPosErrorR.sum() + MaskedPosErrorL.sum()) / TotalHit;

		// 6. Regularização (Manter quats unitários)
		// Usamos os quats ORIGINAIS (sem normalizar) para a penalidade
		auto QuatRegLoss = torch::tensor(0.0f, Input.options());
		std::vector<torch::Tensor> all_quats = {
			PredQuats.slice(1, 0, 4), PredQuats.slice(1, 4, 8), PredQuats.slice(1, 8, 12),
			PredQuats.slice(1, 12, 16), PredQuats.slice(1, 16, 20), PredQuats.slice(1, 20, 24)
		};

		for (auto& q : all_quats)
		{
			auto norm = q.norm(2, 1);
			QuatRegLoss = QuatRegLoss + (norm - 1.0f).pow(2).mean();
		}

		// Como não estamos a elevar ao quadrado, não precisamos de um lambda tão pequeno.
		// Podes voltar a usar um multiplicador normal.
		float lambda_pos = 1.0f;
		float lambda_reg = 0.1f;
		auto TotalLoss = (PosLoss * lambda_pos) + (QuatRegLoss * lambda_reg);
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
