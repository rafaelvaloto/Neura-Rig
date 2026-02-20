// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once

#include "NRCore/NRConcepts.h"
#include "NRCore/NRTypes.h"

namespace NR
{
	template<FloatingPoint T = float>
	/**
	 * @brief Represents a model for handling and predicting INR (Indian Rupee) data.
	 *
	 * This class encapsulates the logic and core functionalities required for
	 * managing INR-related data and performing predictions or computations. It
	 * is the primary interface for interacting with INR-specific operations.
	 */
	class INRModel : public torch::nn::Module
	{
	public:
		~INRModel() override = default;

		/**
		 * @brief Represents a forward operation or entity for processing or calculations.
		 *
		 * @param Input The input tensor for the forward operation.
		 */
		virtual torch::Tensor Forward(torch::Tensor Input) = 0;

		/**
		 * @brief Handles the saving of a trained model to persistent storage.
		 *
		 * This function manages the process of storing a trained model, ensuring
		 * that it is saved in a specific format and location for future use or
		 * deployment. It provides the necessary functionality to persist model
		 * parameters and configurations securely.
		 *
		 * @param FilePath The data structure containing the trained model's parameters and configurations.
		 */
		virtual void SaveModel(const std::string& FilePath) = 0;

		/**
		 * @brief Handles the loading of a pre-trained model from storage.
		 *
		 * This method is responsible for initializing and loading model data
		 * from a specified storage location. It ensures the model is ready for
		 * use in computations or predictions after the loading process.
		 *
		 * @param FilePath The data structure containing the trained model's parameters and configurations.
		 */
		virtual void LoadModel(const std::string& FilePath) = 0;
	};
} // namespace NR
