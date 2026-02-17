// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRNetwork/NRNetwork.h"
#include "NRCore/NRTypes.h"
#include <iostream>

namespace NR
{
	NRNetwork::NRNetwork()
	    : serverSocket(INVALID_SOCKET)
	    , bIsRunning(false)
	{
		WSADATA wsaData;
		WSAStartup(MAKEWORD(2, 2), &wsaData); // Inicia o Windows Sockets
	}

	NRNetwork::~NRNetwork()
	{
		Stop();
		WSACleanup();
	}

	bool NRNetwork::StartServer(int port)
	{
		serverSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
		if (serverSocket == INVALID_SOCKET)
		{
			return false;
		}

		serverAddr.sin_family = AF_INET;
		serverAddr.sin_addr.s_addr = INADDR_ANY; // Escuta em todas as interfaces (localhost)
		serverAddr.sin_port = htons(port);

		if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR)
		{
			return false;
		}

		bIsRunning = true;
		std::cout << "Socket:" << port << std::endl;
		return true;
	}

	int NRNetwork::Receive(std::vector<float>& outBuffer)
	{
		char buffer[8192];
		sockaddr_in clientAddr;
		int clientSize = sizeof(clientAddr);

		int bytesReceived = recvfrom(serverSocket, buffer, sizeof(buffer), 0, (sockaddr*)&clientAddr, &clientSize);

		if (bytesReceived > 0)
		{
			int numFloats = bytesReceived / sizeof(float);
			outBuffer.assign((float*)buffer, (float*)buffer + numFloats);
		}
		return bytesReceived;
	}

	void NRNetwork::Stop()
	{
		if (serverSocket != INVALID_SOCKET)
		{
			closesocket(serverSocket);
			serverSocket = INVALID_SOCKET;
		}
		bIsRunning = false;
	}
} // namespace NR
