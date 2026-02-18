// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"
#include "NRNetwork/NRNetwork.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
	using namespace NR;
	NRRigDescription rigDesc;
	NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		std::cout << "------------------------------" << std::endl;

		while (true)
		{
			int bytes = Network.Receive();
			if (bytes > 0)
			{
				uint8_t header = Network.GetHeader();
				if (header == 1)
				{
					std::vector<uint8_t> bones;
					bones.reserve(bytes);

					Network.GetData(bones);
					if (bones.size() > 0)
					{
						std::cout << "Data received: " << bones.size() << " bytes" << std::endl;
						if (NRParse::ConfigureSkeletonRig(rigDesc, bones.data(), bytes))
						{
							std::cout << "Rig configuration updated!" << std::endl;
							std::cout << "------------------------------" << std::endl;
						}
					}
					else
					{
						std::cout << "No data received!" << std::endl;
					}
				}
				else if (header == 2)
				{
					std::cout << "---------- HEADER (0x02) ---------" << std::endl;
					std::vector<float> data;
					data.reserve(bytes / sizeof(float));

					Network.GetData(data);
					for (int i = 0; i < bytes / sizeof(float); i += 3)
					{
						auto bone = i > 0 ? i / 3 : 0;
						float X = data[i];
						float Y = data[i + 1];
						float Z = data[i + 2];
						std::cout << "Bone[" << rigDesc.BoneMap[bone] << "] -> X: " << X << " Y: " << Y << " Z: " << Z << std::endl;
					}
					std::cout << "------------------------------" << std::endl;
				}
			}
			else
			{
				std::cout << "Ping received package[" << bytes << "] bytesData." << std::endl;
				std::cout << "------------------------------" << std::endl;
			}
		}
	}
	return 0;
}
