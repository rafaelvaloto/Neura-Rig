// Project: NeuralRig
// Copyright (c) 2026 rafae
// All rights reserved.

#include "NRNetwork/NRNetwork.h"
#include <iostream>
#include <string>
#include <vector>

int main()
{
	NR::NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		std::vector<float> data;
		while (true)
		{
			int bytes = Network.Receive(data);
			if (bytes > 0)
			{
				int idx = sizeof(float) * 3;
				float X = data[idx];
				float Y = data[idx + 1];
				float Z = data[idx + 2];

				std::cout << "Bone[] -> X: " << X << " Y: " << Y << " Z: " << Z << std::endl;
				std::cout << "------------------------------" << std::endl;
			}
			else if (bytes < 0)
			{
				std::cerr << "Receive error!" << std::endl;
				break;
			}
			else
			{
				std::cout << "Ping received package[" << bytes << "] bytes." << std::endl;
			}
		}
	}
	else
	{
		std::cerr << "Failed to start server on port " << port << std::endl;
	}
	return 0;
}
