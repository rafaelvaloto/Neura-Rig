// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"
#include "NRNetwork/NRNetwork.h"
#include "NRSolver/NRSolver.h"
#include "NRTrainee/NRTrainee.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#include <Windows.h>
#endif

class NRMultiHeadModel : public NR::INRModel<float> {
public:
	torch::nn::Sequential backbone{nullptr};
	torch::nn::Linear head_r{nullptr}, head_l{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden) {
		backbone = register_module("backbone", torch::nn::Sequential(
			torch::nn::Linear(in_size, hidden),
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			torch::nn::ELU(),
			torch::nn::Linear(hidden, hidden),
			torch::nn::ELU()
		));

		head_r = register_module("head_r", torch::nn::Linear(hidden, 3));  // 1 Vec (foot_r)
		head_l = register_module("head_l", torch::nn::Linear(hidden, 3));  // 1 Vec (foot_l)
	}

	torch::Tensor Forward(torch::Tensor x) override {
		auto feat = backbone->forward(x);
		auto out_r = head_r->forward(feat);
		auto out_l = head_l->forward(feat);
		return torch::cat({out_r, out_l}, 1);
	}

	void SaveModel(const std::string& FilePath) override {
		torch::save(shared_from_this(), FilePath);
	}

	void LoadModel(const std::string& FilePath) override {
		torch::load(backbone, FilePath);
	}
};

bool saveModel = false;
int main()
{
	using namespace NR;
	NRModelProfile activeProfile;
	NRRules activeRules;

	std::shared_ptr<NRSolver> solver = nullptr;

	std::string DataAssetPath = "Tests/Datasets/Foot_IK.json";
	if (!std::filesystem::exists(DataAssetPath))
	{
		DataAssetPath = "../Tests/Datasets/Foot_IK.json"; // Fallback for some build configurations
	}

	if (!NRParse::LoadProfileFromJson(DataAssetPath, activeProfile))
	{
		std::cerr << "Failed to load data asset: " << DataAssetPath << std::endl;
		return 1;
	}

	std::cout << "----------------------------------" << std::endl;
	std::cout << "Profile loaded: " << activeProfile.ProfileName << std::endl;
	std::cout << " -> Input Size: " << activeProfile.GetRequiredInputSize() << std::endl;
	std::cout << " -> Output Size: " << activeProfile.GetRequiredOutputSize() << std::endl;

	auto InputSize = activeProfile.GetRequiredInputSize();
	auto model = std::make_shared<NRMultiHeadModel>(InputSize, 1024);
	std::cout << "Model created!" << std::endl;

	auto trainee = std::make_shared<NRTrainee<float>>(model, activeProfile, activeRules, 1e-4);
	std::cout << "Model trainee configuration!" << std::endl;

	NRNetwork dNetwork;
	int dport = 6004;

	if (!dNetwork.StartServer(dport))
	{
		std::cout << "Failed to start debug server on port " << dport << std::endl;
		return 1;
	}

	NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{

		std::cout << "----------------------------------" << std::endl;
		std::cout << "Server Started!" << std::endl;
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
		std::cout << "Success socket Debug port: " << dport << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		static int frameCounter = 0;
		while (true)
		{
			// Recebe dados do socket principal
			int bytes = Network.Receive();
			if (bytes > 0)
			{
				uint8_t header = Network.GetHeader();
				if (header == 2)
				{
					std::vector<float> data;
					Network.GetData(data);
					if (trainee)
					{
						float loss = trainee->TrainStep(data);
						if (frameCounter++ % 30 == 0)
						{
							std::cout << "----------------------------------" << std::endl;
							std::cout << " Loss: " << loss << std::endl;
							std::cout << " frame counter:" << frameCounter << std::endl;
							std::cout << "----------------------------------" << std::endl;
						}

						if (trainee->IdealTarg.defined())
						{
							std::vector<uint8_t> dSendBuffer;
							dSendBuffer.push_back(0x04); // Header debug - IdealTarg

							// Convertendo o tensor IdealTarg para um array de bytes
							const float* dDataPtr = trainee->IdealTarg.data_ptr<float>();
							size_t dNumElements = trainee->IdealTarg.numel();
							size_t dBytesToCopy = dNumElements * sizeof(float);
							auto* dBytePtr = reinterpret_cast<const uint8_t*>(dDataPtr);

							dSendBuffer.insert(dSendBuffer.end(), dBytePtr, dBytePtr + dBytesToCopy);
							size_t dTotalPayloadSize = dSendBuffer.size();
							dNetwork.Send(dSendBuffer.data(), dTotalPayloadSize);
						}

						if (!solver)
						{
							solver = std::make_shared<NRSolver>(model, activeProfile);
							std::cout << "Solver Created Successfully!" << std::endl;
							std::cout << "=== SWITCHING TO SOLVER MODE ===" << std::endl;
						}

						if (solver)
						{
							std::vector<float> solveInput(InputSize);
							std::memcpy(solveInput.data(), data.data(), InputSize * sizeof(float));

							std::vector<float> predicted = solver->Solve(solveInput);

							std::vector<uint8_t> sendBuffer;
							sendBuffer.push_back(0x03); // Header

							size_t bytesToCopy = predicted.size() * sizeof(float);
							auto* bytePtr = reinterpret_cast<uint8_t*>(predicted.data());

							sendBuffer.insert(sendBuffer.end(), bytePtr, bytePtr + bytesToCopy);
							size_t totalPayloadSize = sendBuffer.size();
							Network.Send(sendBuffer.data(), totalPayloadSize);
						}
					}
				}
			}
			else if (bytes == 0)
			{
				std::cout << "----------------------------------" << std::endl;
				std::cout << "Ping received package[0]" << bytes << " bytes." << std::endl;
				std::cout << "----------------------------------" << std::endl;
			}

			// Recebe dados do socket de debug (opcional, caso queira enviar comandos para o server via 6004)
			int dBytes = dNetwork.Receive();
			if (dBytes > 0)
			{
				uint8_t dHeader = dNetwork.GetHeader();
				// Lógica para tratar comandos de debug recebidos na porta 6004, se necessário.
				// Por enquanto apenas logamos.
				std::cout << "Debug command received on port " << dport << " Header: " << (int)dHeader << std::endl;
			}

			// std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
	return 0;
}
