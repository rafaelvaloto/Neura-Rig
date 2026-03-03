// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include "muParser.h"

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
	struct IKLossResult
	{
		torch::Tensor TotalLoss;
		torch::Tensor PosLoss;
		torch::Tensor RotLoss;
		torch::Tensor RegLoss;
	};

	struct NRDataBlock
	{
		std::string Name;
		int32_t Offset;
		int32_t FloatCount;
	};

	struct NRRule {
		std::string Name;
		std::map<std::string, double> Constants;
		std::map<std::string, std::vector<std::string>> Variables;
		std::map<std::string, std::string> Logic;

		struct Phase {
			std::string Id;
			std::string Condition;
			std::map<std::string, std::string> Formulas;
		};
		std::vector<Phase> Phases;
	};

	struct NRBinding {
		std::string BoneName;
		std::string RuleName;
		std::map<std::string, double> Overrides;
	};

	struct NRModelProfile
	{
		std::string ProfileName;
		std::vector<NRRule> Rules;
		std::vector<NRBinding> Bindings;
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

		[[nodiscard]] NRRule FindRule(const std::string& name) const
		{
			for (const auto& rule : Rules)
			{
				if (rule.Name == name)
					return rule;
			}
			return {};
		}

		[[nodiscard]] NRDataBlock FindOutputBlock(const std::string& name) const
		{
			for (const auto& block : Outputs)
			{
				if (block.Name == name)
					return block;
			}
			return {};
		}

		[[nodiscard]] torch::Tensor GetInputBoneValue(const torch::Tensor& Input, const std::string& name) const
		{
			for (const auto& block : Inputs)
			{
				if (block.Name == name)
				{
					return Input.slice(1, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return torch::Tensor(nullptr);
		}

		[[nodiscard]] torch::Tensor GetOutputBoneValue(const torch::Tensor& Output, const std::string& name) const
		{
			for (const auto& block : Outputs)
			{
				if (block.Name == name)
				{
					return Output.slice(1, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return torch::Tensor(nullptr);
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

		static void Debug(const std::string& Message, const torch::Tensor& DataTensor)
		{
			std::cout << "[NRParse] " << Message << " => " << DataTensor << std::endl;
		}
	};

} // namespace NR
