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

void printBuffer(std::vector<float> buffer)
{
	// int offset = 0;
	// for (auto Block : activeProfile.Inputs)
	// {
	// 	std::cout << "-----------------INPUTS----------------" << std::endl;
	// 	std::cout << "-------------"<< offset <<"------------" << std::endl;
	// 	if (Block.FloatCount == 4)
	// 	{
	// 		std::cout << Block.Name << ": x " << data[offset] << " y " << data[offset + 1] << " z " << data[offset+2] << " w " << data[offset+3] << std::endl;
	// 	}
	// 	else if (Block.FloatCount == 3)
	// 	{
	// 		std::cout << Block.Name << ": x " << data[offset] << " y " << data[offset+1] << " z " << data[offset+2] << std::endl;
	// 	}
	// 	else if (Block.FloatCount == 1)
	// 	{
	// 		std::cout << Block.Name << ": " << data[offset] << std::endl;
	// 	}
	// 	offset += Block.FloatCount;
	// 	std::cout << "----------------------------------------" << std::endl;
	// }

	// int offsetT = 57;
	// for (auto Block : activeProfile.Targets)
	// {
	// 	std::cout << "----------------TARGETS----------------" << std::endl;
	// 	std::cout << "-------------"<< offsetT <<"------------" << std::endl;
	// 	if (Block.FloatCount == 4)
	// 	{
	// 		std::cout << Block.Name << ": x " << data[offsetT] << " y " << data[offsetT + 1] << " z " << data[offsetT+2] << " w " << data[offsetT+3] << std::endl;
	// 	}
	// 	else if (Block.FloatCount == 3)
	// 	{
	// 		std::cout << Block.Name << ": x " << data[offsetT] << " y " << data[offsetT+1] << " z " << data[offsetT+2] << std::endl;
	// 	}
	// 	else if (Block.FloatCount == 1)
	// 	{
	// 		std::cout << Block.Name << ": " << data[offsetT] << std::endl;
	// 	}
	// 	offsetT += Block.FloatCount;
	// 	std::cout << "----------------------------------------" << std::endl;
	// }
}

// Definição corrigida do Modelo Multi-Head
class NRMultiHeadModel : public NR::INRModel<float> {
public:
	torch::nn::Sequential backbone{nullptr};
	torch::nn::Linear head_r{nullptr}, head_l{nullptr}, head_p{nullptr}, head_r_pos{nullptr}, head_l_pos{nullptr}, head_p_pos{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden) {
		// Inicializamos e registamos o backbone
		backbone = register_module("backbone", torch::nn::Sequential(
			torch::nn::Linear(in_size, hidden),
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			torch::nn::ELU(),
			torch::nn::Linear(hidden, hidden),
			torch::nn::ELU()
		));

		// Inicializamos e registamos as cabeças
		head_r = register_module("head_r", torch::nn::Linear(hidden, 12)); // 3 Quats (Thigh, Calf, Foot)
		head_l = register_module("head_l", torch::nn::Linear(hidden, 12)); // 3 Quats
		head_p = register_module("head_p", torch::nn::Linear(hidden, 4));  // 1 Quat (Pelvis)
		head_r_pos = register_module("head_r_pos", torch::nn::Linear(hidden, 3));  // 1 Vec (foot_r)
		head_l_pos = register_module("head_l_pos", torch::nn::Linear(hidden, 3));  // 1 Vec (foot_l)
		head_p_pos = register_module("head_p_pos", torch::nn::Linear(hidden, 3));  // 1 Vec (Pelvis)
	}

	// 1. Implementação do Forward (A lógica principal)
	torch::Tensor Forward(torch::Tensor x) override {
		auto feat = backbone->forward(x);

		auto out_r = head_r->forward(feat);
		auto out_l = head_l->forward(feat);
		auto out_p = head_p->forward(feat);
		auto out_r_pos = head_r_pos->forward(feat);
		auto out_l_pos = head_l_pos->forward(feat);
		auto out_p_pos = head_p_pos->forward(feat);

		// Retorna tudo concatenado: [R, L, P] -> Total 28 floats
		return torch::cat({out_r, out_l, out_p, out_r_pos, out_l_pos, out_p_pos}, 1);
	}

	void SaveModel(const std::string& FilePath) override {
		torch::save(shared_from_this(), FilePath);
	}

	void LoadModel(const std::string& FilePath) override {
		torch::load(backbone, FilePath);
	}
};

class NRMLPModel : public NR::INRModel<float>
{
	torch::nn::Sequential Network;

public:
	NRMLPModel(int64_t InputSize, int64_t HiddenSize, int64_t OutputSize)
	{
		Network = torch::nn::Sequential(
			torch::nn::Linear(InputSize, HiddenSize),
			torch::nn::ReLU(),
			torch::nn::Linear(HiddenSize, HiddenSize),
			torch::nn::ReLU(),
			torch::nn::Linear(HiddenSize, OutputSize));
		register_module("Network", Network);
	}

	torch::Tensor Forward(torch::Tensor Input) override
	{
		return Network->forward(Input);
	}

	void SaveModel(const std::string& FilePath) override
	{
		torch::save(Network, FilePath);
	}

	void LoadModel(const std::string& FilePath) override
	{
		torch::load(Network, FilePath);
	}
};

bool saveModel = false;

int main()
{
	using namespace NR;
	NRModelProfile activeProfile;

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
	std::cout << " -> Targets Size: " << activeProfile.GetRequiredTargetsSize() << std::endl;
	std::cout << " -> Output Size: " << activeProfile.GetRequiredOutputSize() << std::endl;

	auto InputSize = activeProfile.GetRequiredInputSize() + activeProfile.GetRequiredTargetsSize();
	auto model = std::make_shared<NRMultiHeadModel>(InputSize, 512);
	std::cout << "Model created!" << std::endl;

	auto trainee = std::make_shared<NRTrainee<float>>(model, activeProfile, 1e-4);
	std::cout << "Model trainee configuration!" << std::endl;

	NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{
		std::cout << "----------------------------------" << std::endl;
		std::cout << "Server Started!" << std::endl;
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
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

						if (loss < 0.1f)
						{
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

								// 2. (24 * 4 = 96)
								size_t bytesToCopy = predicted.size() * sizeof(float);
								auto* bytePtr = reinterpret_cast<uint8_t*>(predicted.data());

								sendBuffer.insert(sendBuffer.end(), bytePtr, bytePtr + bytesToCopy);
								size_t totalPayloadSize = sendBuffer.size(); // 97 bytes
								Network.Send(sendBuffer.data(), totalPayloadSize);
							}
						}

					}
				}
			}
			else
			{
				std::cout << "----------------------------------" << std::endl;
				std::cout << "Ping received package[0]" << bytes << " bytes." << std::endl;
				std::cout << "----------------------------------" << std::endl;
			}
		}
	}
	return 0;
}
