// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include "NRCore/NRTypes.h"
#include "NRInterfaces/INRModel.h"

namespace NR
{
	/**
	 * @brief Solver class that uses a neural network model to compute rig transformations.
	 *
	 * This class wraps a neural network model and provides functionality to solve
	 * inverse kinematics or other rig-related problems using machine learning.
	 */
	class NRSolver
	{
	public:
		/**
		 * @brief Constructs a NRSolver with the given neural network model.
		 * @param ModelNetwork Unique pointer to the neural network model implementation that will be used for solving.
		 * @param DeviceTarget The target device (CPU or CUDA) where tensor computations will be executed. Defaults to CPU.
		 */
		explicit NRSolver(std::unique_ptr<INRModel<float>> ModelNetwork, const torch::Device DeviceTarget = torch::kCPU)
		    : NeuralNetwork(std::move(ModelNetwork))
		    , Device(DeviceTarget)
		{
		}

		/**
		 * @brief Solves the rig problem for the given 3D coordinates.
		 * @param x The X coordinate in 3D space.
		 * @param y The Y coordinate in 3D space.
		 * @param z The Z coordinate in 3D space.
		 */
		void solve(float x, float y, float z);

	private:
		/**
		 * @brief Applies the computed angles to the rig.
		 * @param Angles Tensor containing the joint angles to be applied.
		 */
		static void ApplyToRig(torch::Tensor Angles);

		/**
		 * @brief Unique pointer to the neural network model used for solving.
		 */
		std::unique_ptr<INRModel<float>> NeuralNetwork;

		/**
		 * @brief Device where tensor computations will be executed (CPU or CUDA).
		 */
		torch::Device Device;
	};

} // namespace NR
