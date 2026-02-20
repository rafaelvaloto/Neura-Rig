// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
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
		 * @brief Receives a packet from the network and stores it in the internal buffer.
		 *
		 * Non-blocking operation that attempts to read available data from the socket.
		 * Updates internal header and payloadSize members upon successful reception.
		 *
		 * @return Number of bytes received, or negative value on error.
		 */
		int Receive();

		void Send(const uint8_t* data, int size);

		/**
		 * @brief Extracts typed data from the received payload buffer.
		 *
		 * Copies and converts the internal buffer data into the specified type vector.
		 * Must be called after a successful Receive() operation.
		 *
		 * @tparam T The target data type (typically float for bone transforms).
		 * @param OutBuffer Output vector to populate with parsed data.
		 */
		template<typename T>
		void GetData(std::vector<T>& OutBuffer);

		/**
		 * @brief Returns the header byte from the last received packet.
		 *
		 * Used to identify packet type or protocol version before processing payload.
		 *
		 * @return The 8-bit header value from the most recent Receive() call.
		 */
		uint8_t GetHeader() const;

		/**
		 * @brief Stops the server and closes the socket connection.
		 *
		 * Safe to call multiple times.
		 */
		void Stop();

	private:
		/**
		 * @brief Internal buffer for receiving network data.
		 *
		 * Stores up to 32KB of incoming UDP/TCP packet data before parsing
		 * into structured format. Size must accommodate maximum expected payload.
		 */
		char inBuffer[32768] = {};

		/**
		 * @brief Packet header identifying the data type or protocol version.
		 *
		 * First byte of received packets, used to validate and route incoming data.
		 */
		uint8_t header{};

		/**
		 * @brief Size in bytes of the actual payload data.
		 *
		 * Excludes header bytes, represents the length of valid data in inBuffer.
		 */
		int payloadSize{};

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
		sockaddr_in serverAddr{};

		/**
		 * @brief Stores the address information of the connected client.
		 *
		 * Holds the network address details necessary for communication
		 * with the client during a socket connection.
		 */
		sockaddr_in clientAddr{};

		/**
		 * @brief Indicates whether the server is currently active.
		 *
		 * Set to true after StartServer() succeeds, false after Stop() is called.
		 */
		bool bIsRunning;
	};

} // namespace NR
