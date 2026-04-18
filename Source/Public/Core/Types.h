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
	struct Vec3
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
	};

	static Vec3 operator+(const Vec3& a, const Vec3& b)
	{
		return {a.x + b.x, a.y + b.y, a.z + b.z};
	}

	static Vec3 operator-(const Vec3& a, const Vec3& b)
	{
		return {a.x - b.x, a.y - b.y, a.z - b.z};
	}

	static Vec3 operator*(const Vec3& v, float s)
	{
		return {v.x * s, v.y * s, v.z * s};
	}

	static float Length(const Vec3& v)
	{
		return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
	}

	struct Mat3
	{
		float m[3][3]{};

		static Mat3 Identity()
		{
			Mat3 r;
			r.m[0][0] = 1.0f;
			r.m[0][1] = 0.0f;
			r.m[0][2] = 0.0f;
			r.m[1][0] = 0.0f;
			r.m[1][1] = 1.0f;
			r.m[1][2] = 0.0f;
			r.m[2][0] = 0.0f;
			r.m[2][1] = 0.0f;
			r.m[2][2] = 1.0f;
			return r;
		}
	};

	static Mat3 Multiply(const Mat3& a, const Mat3& b)
	{
		Mat3 r{};
		for (int i = 0; i < 3; ++i)
		{
			for (int j = 0; j < 3; ++j)
			{
				r.m[i][j] = 0.0f;
				for (int k = 0; k < 3; ++k)
					r.m[i][j] += a.m[i][k] * b.m[k][j];
			}
		}
		return r;
	}

	static Vec3 Multiply(const Mat3& m, const Vec3& v)
	{
		return {
			m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
			m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
			m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z
		};
	}

	static Mat3 RotX(float a)
	{
		const float c = std::cos(a);
		const float s = std::sin(a);
		return Mat3{{
			{1, 0, 0},
			{0, c, -s},
			{0, s, c}
		}};
	}

	static Mat3 RotY(float a)
	{
		const float c = std::cos(a);
		const float s = std::sin(a);
		return Mat3{{
			{c, 0, s},
			{0, 1, 0},
			{-s, 0, c}
		}};
	}

	static Mat3 RotZ(float a)
	{
		const float c = std::cos(a);
		const float s = std::sin(a);
		return Mat3{{
			{c, -s, 0},
			{s, c, 0},
			{0, 0, 1}
		}};
	}

	static Mat3 EulerXYZ(float x, float y, float z)
	{
		// Ordem: X -> Y -> Z
		return Multiply(Multiply(RotX(x), RotY(y)), RotZ(z));
	}

	static float DegToRad(float deg)
	{
		return deg * 3.14159265358979323846f / 180.0f;
	}

	struct Quat
	{
		float x, y, z, w;

		static Quat FromEulerXYZ(float x, float y, float z)
		{
			float cx = std::cos(x * 0.5f);
			float sx = std::sin(x * 0.5f);
			float cy = std::cos(y * 0.5f);
			float sy = std::sin(y * 0.5f);
			float cz = std::cos(z * 0.5f);
			float sz = std::sin(z * 0.5f);

			Quat q;
			q.w = cx * cy * cz + sx * sy * sz;
			q.x = sx * cy * cz - cx * sy * sz;
			q.y = cx * sy * cz + sx * cy * sz;
			q.z = cx * cy * sz - sx * sy * cz;
			return q;
		}

		// Unreal Engine: FRotator(Pitch, Yaw, Roll) → q = q_Yaw(Z) * q_Pitch(Y) * q_Roll(X)
		static Quat FromUnrealRotator(float pitch, float yaw, float roll)
		{
			float cr = std::cos(roll  * 0.5f);
			float sr = std::sin(roll  * 0.5f);
			float cp = std::cos(pitch * 0.5f);
			float sp = std::sin(pitch * 0.5f);
			float cy = std::cos(yaw   * 0.5f);
			float sy = std::sin(yaw   * 0.5f);

			// q = q_Yaw * q_Pitch * q_Roll
			Quat q;
			q.w =  cy * cp * cr + sy * sp * sr;
			q.x =  cy * cp * sr - sy * sp * cr;
			q.y =  cy * sp * cr + sy * cp * sr;
			q.z =  sy * cp * cr - cy * sp * sr;
			return q;
		}
	};

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

	struct NRRule
	{
		struct RotationLimit
		{
			float MinX = -360.0f;
			float MaxX = 360.0f;
			float MinY = -360.0f;
			float MaxY = 360.0f;
			float MinZ = -360.0f;
			float MaxZ = 360.0f;

			[[nodiscard]] bool IsZero() const {
				return MinX == 0.0f && MaxX == 0.0f && MinY == 0.0f && MaxY == 0.0f && MinZ == 0.0f && MaxZ == 0.0f;
			}
		};

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
			struct Pose
			{
				float x = 0, y = 0, z = 0;
				float q1 = 0, q2 = 0, q3 = 0, qw = 1;
			} RestPose;
			NRRule::RotationLimit Limits;
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
