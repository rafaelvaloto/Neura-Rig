// Project: NeuralRig
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
#include "Types.h"

namespace NR
{
	class Parse
	{
	public:
		/**
		 * @brief Loads a model profile from a JSON file.
		 * @param FilePath Path to the JSON file
		 * @param OutProfile Reference to the profile structure to be populated
		 * @return true if successful, false otherwise
		 */
		static bool LoadProfileFromJson(const std::string& FilePath, NRModelProfile& OutProfile);

		static bool LoadIKFromJson(const std::string& FilePath, NRModelProfile& OutProfile);
		static bool LoadSKFromJson(const std::string& FilePath, NRSkeleton& OutSkeleton);
		static bool LoadTWFromJson(const std::string& FilePath, NRTrainingWeights& OutWeights);
	};
} // namespace NR
