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
		WSAStartup(MAKEWORD(2, 2), &wsaData); // Initialize Windows Sockets
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
		serverAddr.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces (localhost)
		serverAddr.sin_port = htons(port);

		if (bind(serverSocket, reinterpret_cast<sockaddr*>(&serverAddr), sizeof(serverAddr)) == SOCKET_ERROR)
		{
			return false;
		}

		bIsRunning = true;
		return true;
	}

	template<typename T>
	void NRNetwork::GetData(std::vector<T>& OutBuffer)
	{
		if constexpr (std::is_same_v<T, float>)
		{
			if (payloadSize > 0)
			{
				const auto count = (payloadSize / sizeof(float));
				auto data = reinterpret_cast<const float*>(&inBuffer[1]);
				OutBuffer.assign(data, data + count);
			}
		}
		else if constexpr (std::is_same_v<T, uint8_t>)
		{
			if (payloadSize > 0)
			{
				auto data = reinterpret_cast<const uint8_t*>(&inBuffer[1]);
				OutBuffer.assign(data, data + payloadSize);
			}
		}
	}

	uint8_t NRNetwork::GetHeader() const
	{
		return header;
	}

	int NRNetwork::Receive()
	{
		int clientSize = sizeof(clientAddr);
		const int bytesReceived = recvfrom(serverSocket, inBuffer, sizeof(inBuffer), 0, reinterpret_cast<sockaddr*>(&clientAddr), &clientSize);

		if (bytesReceived == SOCKET_ERROR)
		{
			int err = WSAGetLastError();
			if (err != WSAEWOULDBLOCK)
			{
				std::cerr << "[NRNetwork] Error receiving data: " << err << std::endl;
			}
			return -1;
		}

		header = 0;
		payloadSize = 0;
		if (bytesReceived > 1)
		{
			header = inBuffer[0];
			payloadSize = bytesReceived - 1;
		}
		return payloadSize;
	}

	void NRNetwork::Send(const uint8_t* data, int size)
	{
		constexpr int clientSize = sizeof(clientAddr);
		sendto(serverSocket, reinterpret_cast<const char*>(data), size, 0, reinterpret_cast<struct sockaddr*>(&clientAddr), clientSize);
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

	template void NRNetwork::GetData<float>(std::vector<float>&);
	template void NRNetwork::GetData<uint8_t>(std::vector<uint8_t>&);
} // namespace NR
