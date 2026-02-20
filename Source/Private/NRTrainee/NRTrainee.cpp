// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T>> TargetModel, NRModelProfile Rig, const double LearningRate)
	    : TargetModel(TargetModel)
	    , RigDesc(Rig)
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));
	}

	template<FloatingPoint T>
	float NRTrainee<T>::TrainStep(const std::vector<NRVector3D>& InputVectors, const std::vector<NRVector3D>& TargetVectors)
	{
		int32_t InCount = RigDesc.GetRequiredInputSize();
		int32_t OutCount = RigDesc.GetRequiredOutputSize();

		int32_t InBatchSize = (InputVectors.size() * 3) / InCount;
		int32_t TargetBatchSize = (TargetVectors.size() * 3) / OutCount;

		at::Tensor InputTensor = torch::from_blob((void*)InputVectors.data(), {InBatchSize, InCount}, torch::kFloat).clone();
		const auto TargetTensor = torch::from_blob((void*)TargetVectors.data(), {TargetBatchSize, OutCount}, torch::kFloat).clone();

		Optimizer->zero_grad();
		const torch::Tensor prediction = TargetModel->Forward(InputTensor);
		const torch::Tensor Loss = torch::mse_loss(prediction, TargetTensor);

		Loss.backward();
		torch::nn::utils::clip_grad_norm_(TargetModel->parameters(), 1.0);
		Optimizer->step();
		return Loss.item<float>();
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
