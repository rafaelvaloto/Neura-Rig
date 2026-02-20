// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)
#pragma warning(disable : 4702)
#pragma warning(disable : 4100)
#include <torch/torch.h>
#pragma warning(pop)

namespace NR
{
	struct NRVector3D
	{
		float x, y, z;
	};

	enum class PacketType : uint8_t
	{
		RigSetup = 0x01,
		BoneData = 0x02
	};

	// Define o que é cada pedaço do seu pacote UDP
	struct NRDataBlock
	{
		std::string Name;   // Ex: "Velocity", "Input", "foot_r"
		int32_t FloatCount; // Ex: 3 (Vector), 2 (Input Axis), 1 (Scalar)
		bool bIsTarget;     // false = Input da IA | true = Output da IA (Label)
	};

	// O seu novo "Template" de Configuração
	struct NRModelProfile
	{
		std::string ProfileName; // Ex: "Locomotion_LowerBody"
		std::vector<NRDataBlock> Inputs;
		std::vector<NRDataBlock> Outputs;

		void AddInput(const std::string& name, int32_t size)
		{
			Inputs.push_back({name, size, false});
		}

		void AddOutput(const std::string& name, int32_t size)
		{
			Outputs.push_back({name, size, true});
		}

		[[nodiscard]] int32_t GetRequiredInputSize() const
		{
			int32_t totalSize = 0;
			for (const auto& block : Inputs)
			{
				totalSize += block.FloatCount;
			}
			return totalSize;
		}

		[[nodiscard]] int32_t GetRequiredOutputSize() const
		{
			int32_t totalSize = 0;
			for (const auto& block : Outputs)
			{
				totalSize += block.FloatCount;
			}
			return totalSize;
		}
	};

} // namespace NR
