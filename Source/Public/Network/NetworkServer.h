// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

#pragma once
#include <winsock2.h>
#include <vector>

#pragma comment(lib, "ws2_32.lib")

namespace NR
{
	/**
	 * @brief Simplified UDP server to receive animation data (floats).
	 */
	class NetworkServer
	{
	public:
		NetworkServer();
		~NetworkServer();

		/**
		 * @brief Starts the server on a specific port.
		 * @param port Port to listen on.
		 * @return true if started successfully.
		 */
		bool Start(int port);

		/**
		 * @brief Receives data from the socket. Does not validate headers.
		 * @param outData Vector where received data will be stored.
		 * @return true if data was received.
		 */
		bool Receive(std::vector<float>& outData);

		void Stop();

		bool IsRunning() const { return bIsRunning; }

	private:
		SOCKET serverSocket;
		sockaddr_in serverAddr{};
		bool bIsRunning;
		float buffer[8192]; // Fixed size buffer for performance
	};
}
