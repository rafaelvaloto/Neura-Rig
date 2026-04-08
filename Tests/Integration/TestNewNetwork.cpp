#include "Network/NetworkServer.h"
#include "Network/NetworkClient.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>

int main()
{
    NR::NetworkServer server;
    NR::NetworkClient client;

    const int testPort = 8080;
    
    if (!server.Start(testPort))
    {
        std::cerr << "Failed to start server on port " << testPort << std::endl;
        return 1;
    }

    std::cout << "Server started on port " << testPort << std::endl;

    std::vector<float> dataToSend = { 1.0f, 2.0f, 3.5f, 4.2f };
    
    // Small delay to ensure the socket is ready
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    if (!client.Send(dataToSend, "127.0.0.1", testPort))
    {
        std::cerr << "Failed to send data" << std::endl;
        return 1;
    }

    std::cout << "Data sent. Waiting for reception..." << std::endl;

    std::vector<float> receivedData;
    bool success = false;
    for (int i = 0; i < 10; ++i)
    {
        if (server.Receive(receivedData))
        {
            success = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (success)
    {
        std::cout << "Data received successfully! Count: " << receivedData.size() << std::endl;
        for (float f : receivedData)
        {
            std::cout << f << " ";
        }
        std::cout << std::endl;

        assert(receivedData.size() == dataToSend.size());
        for (size_t i = 0; i < receivedData.size(); ++i)
        {
            assert(receivedData[i] == dataToSend[i]);
        }
        std::cout << "Validation completed!" << std::endl;
    }
    else
    {
        std::cerr << "Failed to receive data (timeout)" << std::endl;
        return 1;
    }

    server.Stop();
    return 0;
}
