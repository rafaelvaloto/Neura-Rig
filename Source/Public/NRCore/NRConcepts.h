// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include <vector>

namespace NeuraRig
{
	template<typename T>
	concept IsRigInput = requires(T a) {
		{ a.ToTensorData() } -> std::convertible_to<std::vector<float>>;
	};

	template<typename T>
	concept IsRigOutput = requires(std::vector<float> data) {
		{ T::FromTensorData(data) } -> std::same_as<T>;
	};

	class NRConcepts
	{
	};
} // namespace NeuraRig
