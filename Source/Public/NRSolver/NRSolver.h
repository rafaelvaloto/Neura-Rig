// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
		 * @param Inputs Vector of input floats (positions, quaternions, etc.) as defined in the profile.
		 * @return Vector of computed rig transformation floats corresponding to the outputs defined in the profile.
		 */
		std::vector<float> Solve(const std::vector<float>& Inputs);

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
