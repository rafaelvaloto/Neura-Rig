// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"
#include "NRNetwork/NRNetwork.h"
#include <iostream>
#include <string>
#include <vector>
// Definições de cores ANSI
#define RESET   "\033[0m"
#define GREEN   "\033[32m"      /* Verde para Convergência */
#define YELLOW  "\033[33m"      /* Amarelo para Logs de Treino */
#define RED     "\033[31m"      /* Vermelho para Loss Alta */
#define CYAN    "\033[36m"      /* Ciano para Headers */

#include "NRTrainee/NRTrainee.h"


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


int main()
{
	using namespace NR;
	NRRigDescription rigDesc;
	std::shared_ptr<NRTrainee<float>> trainee = nullptr;

	NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		std::cout << "------------------------------" << std::endl;

		while (true)
		{
			int bytes = Network.Receive();
			if (bytes > 0)
			{
				uint8_t header = Network.GetHeader();
				if (header == 1)
				{
					std::vector<uint8_t> bones;
					bones.reserve(bytes);

					Network.GetData(bones);
					if (bones.size() > 0)
					{
						std::cout << "Data received: " << bones.size() << " bytes" << std::endl;
						if (NRParse::ConfigureSkeletonRig(rigDesc, bones.data(), bytes))
						{
							std::cout << "Rig configuration updated!" << std::endl;
							std::cout << "------------------------------" << std::endl;

							auto model = std::make_shared<NRMLPModel>(
								rigDesc.GetRequiredInputSize(),
								64,
								rigDesc.GetRequiredInputSize()
							);
							trainee = std::make_shared<NRTrainee<float>>(model, rigDesc, 0.001);
							std::cout << "Model trainee configuration!" << std::endl;
							std::cout << "------------------------------" << std::endl;
						}
					}
					else
					{
						std::cout << "No data received!" << std::endl;
					}
				}
				else if (header == 2)
				{
					std::cout << "---------- HEADER (0x02) ---------" << std::endl;
					std::vector<float> data;
					std::vector<NRVector3D> inputBuffer;
					data.reserve(bytes / sizeof(float));

					Network.GetData(data);
					float* rawFloats = &data[0];
					int numVectors = bytes / sizeof(NRVector3D);
					for (int i = 0; i < numVectors; i++) {
						auto bone = i*3 > 0 ? i : 0;
						inputBuffer.push_back({rawFloats[i*3], rawFloats[i*3+1], rawFloats[i*3+2]});
						std::cout << "Bone[" << rigDesc.BoneMap[bone] << "] -> X: " << rawFloats[i*3] << " Y: " << rawFloats[i*3+1] << " Z: " << rawFloats[i*3+2] << std::endl;
					}

					std::vector<NRVector3D> inputFrame = { inputBuffer[0], inputBuffer[1] };
					std::vector<NRVector3D> targetFrame = { inputBuffer[2], inputBuffer[3] };
					float loss = trainee->TrainStep(inputFrame, targetFrame);

					static int frameCounter = 0;
					if (frameCounter++ % 30 == 0) {
						// Usamos Amarelo para o log comum de treino
						std::cout << YELLOW << "[TRAIN]" << RESET
								  << " Frame: " << CYAN << frameCounter << RESET
								  << " | Loss: " << (loss > 0.1 ? RED : GREEN) << loss << RESET << std::endl;
					}

					if (loss < 0.001) {
						// Verde brilhante para quando o modelo atinge a perfeição
						std::cout << GREEN << "!!! Model Converged !!!" << RESET
								  << " Loss: " << GREEN << loss << RESET << std::endl;

						trainee->SaveWeights("rig_model.pt");
						std::cout << CYAN << "-> Weights saved: " << RESET << "rig_model.pt" << std::endl;
					}
					std::cout << "----------------------------------" << std::endl;
				}
			}
			else
			{
				std::cout << "Ping received package[" << bytes << "] bytesData." << std::endl;
				std::cout << "------------------------------" << std::endl;
			}
		}
	}
	return 0;
}
