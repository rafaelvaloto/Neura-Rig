// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T>> TargetModel, NRRigDescription Rig, double LearningRate)
	    : TargetModel(TargetModel)
	    , RigDesc(Rig)
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));
	}

	template<FloatingPoint T>
	float NRTrainee<T>::TrainStep(const std::vector<NRVector3D>& InputVectors, const std::vector<NRVector3D>& TargetVectors)
	{
		int64_t BatchSize = InputVectors.size() / RigDesc.TargetCount;

		at::Tensor InputTensor = torch::from_blob((void*)InputVectors.data(), {BatchSize, RigDesc.GetRequiredInputSize()}, torch::kFloat).clone();

		const auto TargetTensor = torch::from_blob((void*)TargetVectors.data(), {BatchSize, RigDesc.GetRequiredOutputSize()}, torch::kFloat).clone();

		Optimizer->zero_grad();

		torch::Tensor prediction = TargetModel->Forward(InputTensor);
		torch::Tensor Loss = torch::mse_loss(prediction, TargetTensor);

		Loss.backward();
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
