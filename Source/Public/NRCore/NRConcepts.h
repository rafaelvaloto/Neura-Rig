// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include <concepts>
#include <type_traits>

namespace NR
{
	template<typename T>
	concept FloatingPoint = std::is_floating_point_v<T>;

	template<typename T>
	concept IsTransform = requires(T v) {
		{ v.position } -> std::convertible_to<float*>;
		{ v.rotation } -> std::convertible_to<float*>;
	};

	template<typename T>
	concept IsRigStructure = requires(T rig) {
		{ rig.get_bone_count() } -> std::same_as<size_t>;
		{ rig.get_root() };
	};

} // namespace NR
