// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include <utility>

#include <utility>

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
		 * @param Description Configuration for the rig to be solved.
		 * @param DeviceTarget The target device (CPU or CUDA) where tensor computations will be executed. Defaults to CPU.
		 */
		explicit NRSolver(const std::shared_ptr<INRModel<float>>& ModelNetwork, const NRModelProfile& Description, const torch::Device DeviceTarget = torch::kCPU)
		    : NeuralNetwork(ModelNetwork)
		    , RigDesc(Description)
		    , Device(DeviceTarget)
		{
			NeuralNetwork->to(Device);
			NeuralNetwork->eval();
		}

		/**
		 * @brief Solves the rig transformations for the given target positions.
		 * @param Targets Vector of target positions for which the rig transformations need to be computed.
		 * @return Vector of computed rig transformations corresponding to the input targets.
		 */
		std::vector<NRVector3D> Solve(const std::vector<NRVector3D>& Targets);

	private:
		/**
		 * @brief Unique pointer to the neural network model used for solving.
		 */
		std::shared_ptr<INRModel<float>> NeuralNetwork;

		/**
		 * @brief Device where tensor computations will be executed (CPU or CUDA).
		 */
		torch::Device Device;

		/**
		 * @brief Data structure that describes the configuration and properties of a rig.
		 *
		 * This structure encapsulates the essential details required to define a rig's
		 * components, hierarchy, and operational parameters. It serves as the foundational
		 * descriptor for setting up and manipulating rig-based systems.
		 */
		NRModelProfile RigDesc;
	};

} // namespace NR
