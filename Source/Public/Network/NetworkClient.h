// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto

#pragma once
#include <winsock2.h>
#include <vector>
#include <string>

#pragma comment(lib, "ws2_32.lib")

namespace NR
{
	/**
	 * @brief Simplified UDP client to send animation data (floats).
	 */
	class NetworkClient
	{
	public:
		NetworkClient();
		~NetworkClient();

		/**
		 * @brief Sends a vector of floats to a specific address and port.
		 * @param data Data to be sent.
		 * @param ip Destination IP address (e.g., "127.0.0.1").
		 * @param port Destination port.
		 * @return true if sent successfully.
		 */
		bool Send(const std::vector<float>& data, const std::string& ip, int port);

	private:
		SOCKET clientSocket;
	};
}
