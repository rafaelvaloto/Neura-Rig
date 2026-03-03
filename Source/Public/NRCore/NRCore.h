// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include <type_traits>

namespace NR
{
	template<typename T>
	concept FloatingPoint = std::is_floating_point_v<T>;

} // namespace NR
