// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include "NRTypes.h"

namespace NR
{
	class NRParse
	{
	public:
		/**
		 * @brief Configures a skeleton rig from raw binary data.
		 *
		 * Parses the provided byte array to populate the rig description structure
		 * with bone hierarchy, constraints, and other skeletal configuration data.
		 *
		 * @param rig Reference to the rig description structure to be populated
		 * @param data Pointer to the raw binary data containing rig configuration
		 * @param totalBytes Total size of the data buffer in bytes
		 * @return true if the rig was successfully configured, false otherwise
		 */
		static bool ConfigureSkeletonRig(NRRigDescription& rig, uint8_t* data, int totalBytes);
	};
} // namespace NR
