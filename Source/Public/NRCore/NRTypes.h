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
		std::string Name;
		int32_t Offset;
		int32_t FloatCount;
	};

	// Forward Kinematics de uma chain de 2 bones
	// Recebe: hip_pos(B,3), bone_lengths(L1, L2), 3 quats(B,4 cada)
	// Retorna: foot_pos calculado(B,3)
	struct FKResult
	{
		torch::Tensor TopPos_P;   // (B, 3)
		torch::Tensor Offset_x;  // (B, 3) - útil para regularização
	};

	struct IKLossResult
	{
		torch::Tensor TotalLoss;
		torch::Tensor PosLoss;
		torch::Tensor RotLoss;
		torch::Tensor RegLoss;
	};

	struct AnalyticResult {
		torch::Tensor ThighQuat;
		torch::Tensor CalfQuat;
	};

	// O seu novo "Template" de Configuração
	struct NRModelProfile
	{
		std::string ProfileName; // Ex: "Locomotion_LowerBody"
		std::vector<NRDataBlock> Inputs;
		std::vector<NRDataBlock> Targets;
		std::vector<NRDataBlock> Outputs;

		torch::Tensor GetInputBoneValue(const torch::Tensor& Input, const std::string& name, bool isTarget = false) const
		{
			std::vector<NRDataBlock> DataBlocks = isTarget ? Targets : Inputs;
			for (const auto& block : DataBlocks)
			{
				if (block.Name == name)
				{
					return Input.slice(1, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return torch::Tensor();
		}

		torch::Tensor GetOutputBoneValue(const torch::Tensor& Output, const std::string& name) const
		{
			for (const auto& block : Outputs)
			{
				if (block.Name == name)
				{
					return Output.slice(1, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return torch::Tensor();
		}

		void AddInput(const std::string& name, int32_t size)
		{
			Inputs.push_back({name, size, false});
		}

		void AddTarget(const std::string& name, int32_t size)
		{
			Targets.push_back({name, size, false});
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

		[[nodiscard]] int32_t GetRequiredTargetsSize() const
		{
			int32_t totalSize = 0;
			for (const auto& block : Targets)
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

		void Debug(std::string& Message, const torch::Tensor& DataTensor)
		{
			std::cout << "[NRParse] " << Message << " => " << DataTensor << std::endl;
		}
	};



} // namespace NR
