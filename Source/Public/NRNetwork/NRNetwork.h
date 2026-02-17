// Project: NeuralRig
// Copyright (c) 2026 rafae
// All rights reserved.
#pragma once
#include "NRCore/NRTypes.h"
#include <vector>
#include <winsock2.h>

#pragma comment(lib, "ws2_32.lib")

namespace NR
{
	/**
	 * @brief Network receiver for neural rig procedural data transmission.
	 *
	 * Provides a UDP/TCP socket server to receive floating-point arrays
	 * from external animation systems or training pipelines.
	 */
	class NRNetwork
	{
	public:
		/**
		 * @brief Constructs the network interface without initializing sockets.
		 *
		 * Call StartServer() to bind and listen on a specific port.
		 */
		NRNetwork();

		/**
		 * @brief Destroys the network interface and releases socket resources.
		 *
		 * Automatically calls Stop() if the server is still running.
		 */
		~NRNetwork();

		/**
		 * @brief Initializes and binds the server socket to the specified port.
		 *
		 * @param port The UDP/TCP port to listen on (default: 6003).
		 * @return true if the server started successfully, false otherwise.
		 */
		bool StartServer(int port = 6003);

		/**
		 * @brief Receives a buffer of floating-point data from the network.
		 *
		 * Blocks until data is available or an error occurs.
		 *
		 * @param outBuffer Output vector to store received float values.
		 * @return Number of bytes received, or -1 on error.
		 */
		int Receive(std::vector<float>& outBuffer);

		/**
		 * @brief Stops the server and closes the socket connection.
		 *
		 * Safe to call multiple times.
		 */
		void Stop();

	private:
		/**
		 * @brief Socket handle for the server connection.
		 *
		 * Represents the underlying WinSock socket used for UDP/TCP communication.
		 */
		SOCKET serverSocket;

		/**
		 * @brief Server address configuration structure.
		 *
		 * Contains the IP address, port, and protocol family information
		 * for the bound socket endpoint.
		 */
		sockaddr_in serverAddr;

		/**
		 * @brief Indicates whether the server is currently active.
		 *
		 * Set to true after StartServer() succeeds, false after Stop() is called.
		 */
		bool bIsRunning;
	};
} // namespace NR
