// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
	torch::nn::Linear head_foot_ik{nullptr};
	torch::nn::Linear head_pelvis_ik{nullptr};
	torch::nn::Linear head_spine_ik{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden, int64_t out_size) {
		backbone = register_module("backbone", torch::nn::Sequential(
			torch::nn::Linear(in_size, hidden),
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			torch::nn::ELU(),
			torch::nn::Linear(hidden, hidden),
			torch::nn::ELU()
		));

		head_foot_ik = register_module("head_foot_ik", torch::nn::Linear(hidden, 48));
		head_pelvis_ik = register_module("head_pelvis_ik", torch::nn::Linear(hidden, 6));
		head_spine_ik = register_module("head_spine_ik", torch::nn::Linear(hidden, 6));
	}

	torch::Tensor Forward(torch::Tensor x) override {
		auto feat = backbone->forward(x);
		auto h_foots = head_foot_ik->forward(feat);
		auto h_spine = head_spine_ik->forward(feat);
		auto h_pelvis = head_pelvis_ik->forward(feat);

		return torch::cat({h_foots, h_pelvis, h_spine}, 1);
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
	auto out_size = activeProfile.GetRequiredOutputSize();
	auto model = std::make_shared<NRMultiHeadModel>(InputSize, 256, out_size);
	std::cout << "Model created!" << std::endl;

	auto trainee = std::make_shared<NRTrainee<float>>(model, activeProfile, activeRules, 4e-3);
	std::cout << "Model trainee configuration!" << std::endl;

	NRNetwork dNetwork;
	int dport = 6004;

	if (!dNetwork.StartServer(dport))
	{
		std::cout << "Failed to start debug server on port " << dport << std::endl;
		trainee->Reset();
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


						if (trainee->IdealTargets.defined())
						{
							std::vector<uint8_t> dSendBuffer;
							dSendBuffer.push_back(0x04); // Header debug - IdealTarg

							// Convertendo o tensor IdealTarg para um array de bytes
							const float* dDataPtr = trainee->IdealTargets.data_ptr<float>();
							auto dNumElements = trainee->IdealTargets.numel();
							auto dBytesToCopy = dNumElements * sizeof(float);
							auto* dBytePtr = reinterpret_cast<const uint8_t*>(dDataPtr);

							dSendBuffer.insert(dSendBuffer.end(), dBytePtr, dBytePtr + dBytesToCopy);
							dNetwork.Send(dSendBuffer.data(), dSendBuffer.size());
						}

						// if (trainee->Predicated.defined())
						// {
						// 	std::vector<uint8_t> SendBuffer;
						// 	SendBuffer.push_back(0x03); // Header debug - IdealTarg
						//
						// 	// Convertendo o tensor IdealTarg para um array de bytes
						// 	const float* DataPtr = trainee->Predicated.data_ptr<float>();
						// 	size_t NumElements = trainee->Predicated.numel();
						// 	size_t BytesToCopy = NumElements * sizeof(float);
						// 	auto* BytePtr = reinterpret_cast<const uint8_t*>(DataPtr);
						//
						// 	SendBuffer.insert(SendBuffer.end(), BytePtr, BytePtr + BytesToCopy);
						// 	size_t TotalPayloadSize = SendBuffer.size();
						// 	Network.Send(SendBuffer.data(), TotalPayloadSize);
						// }

						if (!solver && loss < 0.00001)
						{
							solver = std::make_shared<NRSolver>(model, activeProfile);
							std::cout << "=== SWITCHING TO SOLVER MODE ===" << std::endl;
						}

						if (solver)
						{
							// std::vector<float> solveInput(InputSize);
							// std::memcpy(solveInput.data(), data.data(), InputSize * sizeof(float));

							std::vector<float> predicted = solver->Solve(data);

							std::vector<uint8_t> sendBuffer;
							sendBuffer.push_back(0x03); // Header

							auto bytesToCopy = predicted.size() * sizeof(float);
							auto* bytePtr = reinterpret_cast<uint8_t*>(predicted.data());

							sendBuffer.insert(sendBuffer.end(), bytePtr, bytePtr + bytesToCopy);
							Network.Send(sendBuffer.data(), sendBuffer.size());
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
				// L�gica para tratar comandos de debug recebidos na porta 6004, se necess�rio.
				// Por enquanto apenas logamos.
				std::cout << "Debug command received on port " << dport << " Header: " << (int)dHeader << std::endl;
			}
		}
	}

	trainee->Reset();
	return 0;
}
