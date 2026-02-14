// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#include "NRTrainee/NRTrainee.h"

namespace NR
{
	template<FloatingPoint T>
	NRTrainee<T>::NRTrainee(std::shared_ptr<INRModel<T>> TargetModel, double LearningRate)
	    : TargetModel(TargetModel)
	{
		Optimizer = std::make_unique<torch::optim::Adam>(TargetModel->parameters(), torch::optim::AdamOptions(LearningRate));
	}

	template<FloatingPoint T>
	float NRTrainee<T>::TrainStep(const std::vector<NRVector3D>& InputVectors, const std::vector<NRVector3D>& TargetVectors)
	{
		at::Tensor InputTensor = torch::from_blob((void*)InputVectors.data(), {static_cast<int64_t>(InputVectors.size()), 3}, torch::kFloat).clone();
		const auto TargetTensor = torch::from_blob((void*)TargetVectors.data(), {static_cast<int64_t>(TargetVectors.size()), 3}, torch::kFloat).clone();

		/// Zeros out the gradients of all parameters.
		Optimizer->zero_grad();

		// 3. Forward: Try to guess.
		torch::Tensor prediction = TargetModel->Forward(InputTensor);

		/// Calculate the error
		torch::Tensor Loss = torch::mse_loss(prediction, TargetTensor);

		/// Computes the gradient of current tensor with respect to graph leaves.
		Loss.backward();

		Optimizer->step();
		return Loss.item<float>();
	}
	template class NRTrainee<float>;
	template class NRTrainee<double>;
} // namespace NR
