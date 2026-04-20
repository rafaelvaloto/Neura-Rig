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

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
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
	static float DegToRad(float deg)
	{
		return deg * 3.14159265358979323846f / 180.0f;
	}

	using Quat = torch::Tensor; // Shape [4] ou [N, 4]
	using Vec3 = torch::Tensor; // Shape [3] ou [N, 3]

	struct FootValidationPair
	{
		torch::Tensor err_loss;
	};

	struct IKLossResult
	{
		torch::Tensor TotalLoss;
		torch::Tensor PositionLoss;
		torch::Tensor KinematicsLoss;
		torch::Tensor TemporalLoss;
		torch::Tensor AccelerationLoss;
		torch::Tensor SmoothOutputLoss;
		torch::Tensor QuaternionNormLoss;
		torch::Tensor FootTargetLoss;
		torch::Tensor RotationLoss;

		IKLossResult() = default;
		IKLossResult(torch::Tensor Total)
			: TotalLoss(std::move(Total)) {}
	};

	struct NRDataBlock
	{
		std::string Name;
		int32_t Offset;
		int32_t FloatCount;
	};

	struct NRFormula
	{
		std::string Name;
		std::string Expr;
	};

	struct NRVars
	{
		std::string Name;
		std::vector<std::string> List;
	};

	struct NRLogic
	{
		std::string Name;
		std::string Expr;
	};

	struct Pose {
		Vec3 Pos; // [Position]
		Quat Rot; // [Rotate]
	};

	struct RotationLimit {
		torch::Tensor Min; // [MinX, MinY, MinZ]
		torch::Tensor Max; // [MaxX, MaxY, MaxZ]
	};

	struct NRRule
	{
		torch::Tensor RestRotationEuler;

		std::string Name;
		std::map<std::string, double> Constants;
		std::vector<NRVars> Variables;
		std::vector<NRLogic> Logic;
		RotationLimit Limits;

		struct Phase
		{
			std::string Id;
			std::string Condition;
			std::vector<NRFormula> Formulas;
		};

		std::vector<Phase> Phases;
	};

	struct NRSkeleton
	{
		struct Bone
		{
			std::string Name;
			int32_t Size = 0;
			int32_t Offset = 0;
			Pose RestPose;
			RotationLimit Limits;
			std::vector<int32_t> ChildrenIndices;
		};

		Bone Parent;
		std::vector<std::vector<Bone>> Rest; // Cadeias de ossos
	};

	struct NRWeight
	{
		float Weight = 0.1f;
		std::string Description;
	};

	struct NRTrainingWeights
	{
		struct HyperParams
		{
			float LearningRate = 0.0001f;
			float EmaAlpha = 0.15f;
			int32_t MaxCandidates = 5;
		} HyperParameters;

		std::unordered_map<std::string, NRWeight> LossWeights;

		struct BoneBias
		{
			std::string Name;
			float PositionMultiplier = 0.1f;
			float RotationMultiplier = 0.1f;
		};
		std::vector<BoneBias> BoneSpecificBias;
	};

	struct NRBinding
	{
		std::string BoneName;
		std::string RuleName;
		std::vector<NRRule> Rules;
		int Size;
		int Offset;
	};

	struct NRModelProfile
	{
		std::string ProfileName;
		std::vector<NRBinding> Bindings;
		std::vector<NRDataBlock> Inputs;
		std::vector<NRDataBlock> Outputs;

		NRSkeleton Skeleton;
		NRTrainingWeights TrainingWeights;

		void AddInput(const std::string& name, int32_t size, int32_t offset)
		{
			Inputs.push_back({name, offset, size});
		}

		void AddOutput(const std::string& name, int32_t size, int32_t offset)
		{
			Outputs.push_back({name, offset, size});
		}

		[[nodiscard]] NRRule FindRule(const std::string& name, int index) const
		{
			for (const auto& rule : Bindings[index].Rules)
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
					// Se o tensor for 2D (Batch), corta na dim 1. Se for 1D (Single), corta na 0.
					int64_t dimToSlice = (Input.dim() > 1) ? 1 : 0;
					return Input.slice(dimToSlice, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return {};
		}

		[[nodiscard]] torch::Tensor GetOutputBoneValue(const torch::Tensor& Output, const std::string& name) const
		{
			for (const auto& block : Outputs)
			{
				if (block.Name == name)
				{
					int64_t dim = (Output.dim() > 1) ? 1 : 0;
					return Output.slice(dim, block.Offset, block.Offset + block.FloatCount);
				}
			}
			return {};
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
