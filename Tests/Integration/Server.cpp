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

#include "NRSolver/NRSolver.h"
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

bool saveModel = false;

int main()
{
	using namespace NR;
	NRRigDescription rigDesc;
	rigDesc.TargetCount = 2;
	std::shared_ptr<NRTrainee<float>> trainee = nullptr;
	std::shared_ptr<NRSolver> solver = nullptr;
	std::shared_ptr<NRMLPModel> model = nullptr;

	NRNetwork Network;
	int port = 6003;
	if (Network.StartServer(port))
	{
		std::cout << "Success socket NeuralRig port: " << port << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		std::cout << "------------------------------" << std::endl;
		static int frameCounter = 0;
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

							model = std::make_shared<NRMLPModel>(
								rigDesc.GetRequiredInputSize(),
								64,
								rigDesc.GetRequiredOutputSize()
							);
							trainee = std::make_shared<NRTrainee<float>>(model, rigDesc, 0.0001);
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
					// std::cout << "---------- HEADER (0x02) ---------" << std::endl;
					std::vector<float> data;
					std::vector<NRVector3D> inputBuffer;
					data.reserve(bytes / sizeof(float));

					Network.GetData(data);
					float* rawFloats = &data[0];
					int numVectors = bytes / sizeof(NRVector3D);
					for (int i = 0; i < numVectors; i++) {
						inputBuffer.push_back({rawFloats[i*3], rawFloats[i*3+1], rawFloats[i*3+2]});
						// std::string color = (i < 2) ? YELLOW : CYAN;
						// std::cout << color << "Bone[" << rigDesc.BoneMap[i] << "]" << RESET
						// 		  << " -> X: " << rawFloats[i*3] << " Y: " << rawFloats[i*3+1] << " Z: " << rawFloats[i*3+2] << std::endl;
					}

					float loss = trainee->TrainStep({ inputBuffer[0], inputBuffer[1] }, { inputBuffer[2], inputBuffer[3] });
					if (frameCounter++ % 30 == 0) {
						std::cout << "----------------------------------" << std::endl;
						std::cout << " Loss: " << loss << std::endl;
						std::cout << " frame counter:" << frameCounter << std::endl;
						std::cout << "----------------------------------" << std::endl;
					}

					if (loss < 0.001 && !saveModel) {
						saveModel = true;
						std::cout << "----------------------------------" << std::endl;
						// Verde brilhante para quando o modelo atinge a perfeição
						std::cout << "!!! Model Converged !!!" << std::endl;
						std::cout << " Loss: " << loss << std::endl;

						trainee->SaveWeights("rig_model.pt");
						std::cout << "-> Weights saved: " << "rig_model.pt" << std::endl;
						std::cout << "----------------------------------" << std::endl;
					}

					if (loss < 0.0001) { // Quando atingir a precisão desejada
						if (!solver) {
							std::cout << CYAN << "\n=== SWITCHING TO SOLVER MODE ===" << RESET << std::endl;

							// Em vez de LoadModel (que pode falhar se o arquivo estiver ocupado),
							// usamos o modelo que já está na memória e pronto.
							solver = std::make_shared<NRSolver>(model, rigDesc);

							std::cout << GREEN << "Solver Created Successfully!" << RESET << std::endl;
						}

						if (solver) {
							// Realizamos a predição
							std::vector<NRVector3D> frame = { inputBuffer[0], inputBuffer[1] };
							std::vector<NRVector3D> predicted = solver->Solve(frame);

							std::vector<uint8_t> sendBuffer;
							sendBuffer.push_back(0x03);
							float rawOutput[6] = {
								predicted[0].x, predicted[0].y, predicted[0].z,
								predicted[1].x, predicted[1].y, predicted[1].z
							};

							uint8_t* bytePtr = reinterpret_cast<uint8_t*>(rawOutput);
							sendBuffer.insert(sendBuffer.end(), bytePtr, bytePtr + sizeof(rawOutput));
							Network.Send(sendBuffer.data(), sendBuffer.size());

							if (frameCounter % 10 == 0) {
								std::cout << "----------------------------------" << std::endl;
								std::cout << "---------------Real---------------" << std::endl;
								std::cout << "X " << inputBuffer[2].x << " Y " << inputBuffer[2].y << " Z " << inputBuffer[2].z << std::endl;
								std::cout << "X " << inputBuffer[3].x << " Y " << inputBuffer[3].y << " Z " << inputBuffer[3].z << std::endl;
								std::cout << "-------------Predicted------------" << std::endl;
								std::cout << "X " << predicted[0].x << " Y " << predicted[0].y << " Z " << predicted[0].z << std::endl;
								std::cout << "X " << predicted[1].x << " Y " << predicted[1].y << " Z " << predicted[1].z << std::endl;
								std::cout << "----------------------------------" << std::endl;
							}
						}
					}

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
