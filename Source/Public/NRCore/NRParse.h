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
		 * @brief Loads a model profile from a JSON file.
		 * @param FilePath Path to the JSON file
		 * @param OutProfile Reference to the profile structure to be populated
		 * @return true if successful, false otherwise
		 */
		static bool LoadProfileFromJson(const std::string& FilePath, NRModelProfile& OutProfile);
	};
} // namespace NR
