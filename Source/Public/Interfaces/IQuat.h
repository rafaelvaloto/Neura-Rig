// Project: NeuralRig
// Copyright (c) 2026 rafaelvaloto
// All rights reserved.
//
// Created by rafaelvaloto on 4/19/2026.
//
#pragma once
#include "Core/Types.h"

namespace NR
{
	class IQuat
	{
	public:
		virtual ~IQuat() = default;

		virtual Quat ToQuat(float pitch, float yaw, float roll) const = 0;

		virtual Vec3 ToEuler(const Quat& q) const = 0;
	};
}
