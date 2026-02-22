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
	std::shared_ptr<NRMLPModel> model = nullptr;
	std::shared_ptr<NRTrainee<float> > trainee = nullptr;

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
	auto OutputSize = activeProfile.GetRequiredOutputSize();
	model = std::make_shared<NRMLPModel>(InputSize, 64, OutputSize);

	trainee = std::make_shared<NRTrainee<float>>(model, activeProfile, 1e-4);
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
						float loss = trainee->TrainStep(data, 45.751953f, 41.705513f, 45.752106f, 41.705494f);
						if (frameCounter++ % 30 == 0)
						{
							std::cout << "----------------------------------" << std::endl;
							std::cout << " Loss: " << loss << std::endl;
							std::cout << " frame counter:" << frameCounter << std::endl;
							std::cout << "----------------------------------" << std::endl;
						}

						if (loss < 0.001 && !saveModel)
						{
							saveModel = true;
							std::cout << "----------------------------------" << std::endl;
							std::cout << "!!! Model Converged !!!" << std::endl;
							std::cout << " Loss: " << loss << std::endl;

							trainee->SaveWeights("rig_model.pt");
							std::cout << "-> Weights saved: " << "rig_model.pt" << std::endl;
							std::cout << "----------------------------------" << std::endl;
						}

						if (loss < 0.001 && saveModel)
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
								std::memcpy(solveInput.data(), data.data(), InputSize);

								std::vector<float> predicted = solver->Solve(solveInput);

								std::vector<uint8_t> sendBuffer;
								sendBuffer.push_back(0x03);

								auto* bytePtr = reinterpret_cast<uint8_t*>(predicted.data());
								auto sendSize = predicted.size();
								sendBuffer.insert(sendBuffer.end(), bytePtr, bytePtr + sendSize);

								auto len = sendBuffer.size();
								Network.Send(sendBuffer.data(), len);
								std::cout << "-------------Predicted Output------------" << std::endl;
								for (size_t i = 0; i < predicted.size(); ++i)
								{
									std::cout << "O" << i << " Value: " << predicted[i] << std::endl;
								}
								std::cout << "----------------------------------" << std::endl;
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
