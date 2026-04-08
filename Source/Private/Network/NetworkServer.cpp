// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto

#include "Network/NetworkServer.h"
#include <iostream>

namespace NR
{
	NetworkServer::NetworkServer()
	    : serverSocket(INVALID_SOCKET)
	    , bIsRunning(false)
	{
		WSADATA wsaData;
		WSAStartup(MAKEWORD(2, 2), &wsaData);
	}

	NetworkServer::~NetworkServer()
	{
		Stop();
		WSACleanup();
	}

	bool NetworkServer::Start(int port)
	{
		serverSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
		if (serverSocket == INVALID_SOCKET)
		{
			return false;
		}

		serverAddr.sin_family = AF_INET;
		serverAddr.sin_addr.s_addr = INADDR_ANY;
		serverAddr.sin_port = htons(port);

		if (bind(serverSocket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR)
		{
			closesocket(serverSocket);
			serverSocket = INVALID_SOCKET;
			return false;
		}

		u_long mode = 1;
		ioctlsocket(serverSocket, FIONBIO, &mode);

		bIsRunning = true;
		return true;
	}

	bool NetworkServer::Receive(std::vector<float>& outData)
	{
		sockaddr_in clientAddr{};
		int clientSize = sizeof(clientAddr);
		const int bytesReceived = recvfrom(serverSocket, reinterpret_cast<char*>(buffer), sizeof(buffer), 0, reinterpret_cast<sockaddr*>(&clientAddr), &clientSize);

		if (bytesReceived <= 0)
		{
			return false;
		}

		const int count = bytesReceived / sizeof(float);
		outData.assign(buffer, buffer + count);
		return true;
	}

	void NetworkServer::Stop()
	{
		if (serverSocket != INVALID_SOCKET)
		{
			closesocket(serverSocket);
			serverSocket = INVALID_SOCKET;
		}
		bIsRunning = false;
	}
}
