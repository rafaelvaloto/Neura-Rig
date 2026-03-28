// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto

#include "NRNetwork/NRNetworkClient.h"
#include <ws2tcpip.h>

namespace NR
{
	NRNetworkClient::NRNetworkClient()
	{
		WSADATA wsaData;
		WSAStartup(MAKEWORD(2, 2), &wsaData);
		clientSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
	}

	NRNetworkClient::~NRNetworkClient()
	{
		if (clientSocket != INVALID_SOCKET)
		{
			closesocket(clientSocket);
		}
		WSACleanup();
	}

	bool NRNetworkClient::Send(const std::vector<float>& data, const std::string& ip, int port)
	{
		if (clientSocket == INVALID_SOCKET || data.empty())
		{
			return false;
		}

		sockaddr_in destAddr{};
		destAddr.sin_family = AF_INET;
		destAddr.sin_port = htons(port);
		inet_pton(AF_INET, ip.c_str(), &destAddr.sin_addr);

		const int bytesSent = sendto(clientSocket, reinterpret_cast<const char*>(data.data()), static_cast<int>(data.size() * sizeof(float)), 0, reinterpret_cast<sockaddr*>(&destAddr), sizeof(destAddr));

		return bytesSent != SOCKET_ERROR;
	}
}
