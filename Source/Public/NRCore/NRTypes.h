// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once

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

	struct NRRigDescription
	{
		// We use the Map to know WHO the bones are
		std::unordered_map<int64_t, std::string> BoneMap;

		// TargetCount is usually what your MODEL expects at output (fixed or configured)
		int64_t TargetCount = 0;

		// Adds a bone and updates the count automatically
		void AddBone(int64_t index, const std::string& name)
		{
			BoneMap[index] = name;
		}

		// O número de ossos capturados é o tamanho do mapa
		int64_t GetBonesCount() const
		{
			return static_cast<int64_t>(BoneMap.size());
		}

		// Input: Baseado nos ossos que recebemos via rede (X, Y, Z para cada um)
		int64_t GetRequiredInputSize() const
		{
			return GetBonesCount() * 3;
		}

		// Output: O que o modelo deve cuspir (geralmente definido pelo modelo carregado)
		int64_t GetRequiredOutputSize() const
		{
			return TargetCount * 3;
		}
	};

} // namespace NR
