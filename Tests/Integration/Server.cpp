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
				if (bytes == 24)
				{
					double receivedVector[3];
					std::memcpy(receivedVector, data.data(), 24);

					double X = receivedVector[0];
					double Y = receivedVector[1];
					double Z = receivedVector[2];

					std::cout << "Message Content -> X: " << X << " Y: " << Y << " Z: " << Z << std::endl;
				}
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
