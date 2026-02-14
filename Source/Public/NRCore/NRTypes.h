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

	struct NRRigDescription
	{
		int64_t BoneCount;
		int64_t TargetCount;

		int64_t GetRequiredInputSize() const { return TargetCount * 3; }
		int64_t GetRequiredOutputSize() const { return BoneCount * 3; }
	};
} // namespace NR
