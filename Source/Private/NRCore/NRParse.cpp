// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"

namespace NR
{

	bool NRParse::ConfigureSkeletonRig(NRRigDescription& rig, uint8_t* data, int size)
	{
		if (size <= 0)
		{
			return false;
		}

		int offset = 0;
		rig.BoneMap.clear();

		// While there is data to read in the buffer
		while (offset + sizeof(int) * 2 <= size)
		{
			// 1. Read Index (4 bytes)
			int32_t boneIndex;
			memcpy(&boneIndex, &data[offset], sizeof(int32_t));
			offset += sizeof(int32_t);

			// 2. Read Name Length (4 bytes)
			int32_t nameLen;
			memcpy(&nameLen, &data[offset], sizeof(int32_t));
			offset += sizeof(int32_t);

			// Safety: Check if name length makes sense
			if (offset + nameLen > size || nameLen <= 0 || nameLen > 256)
			{
				std::cout << "Invalid bone name length: " << nameLen << std::endl;
				break;
			}

			// 3. Read Name (ANSI chars)
			std::string boneName(reinterpret_cast<char*>(&data[offset]), nameLen);
			offset += nameLen;

			// 4. Add to Rig
			rig.AddBone(boneIndex, boneName);
			std::cout << "Mapped Bone: " << boneIndex << " -> " << boneName << std::endl;
		}

		rig.TargetCount = rig.BoneMap.size();
		return true;
	}

} // namespace NR
