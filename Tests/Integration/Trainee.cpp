// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRInterfaces/INRModel.h"
#include "NRTrainee/NRTrainee.h"
#include "NRCore/NRParse.h"
#include <iostream>
#include <filesystem>

class NRMLPModel : public NR::INRModel<float>
{
	torch::nn::Sequential Network;

public:
	NRMLPModel(int64_t InputSize, int64_t HiddenSize, int64_t OutputSize)
	{
		std::cout << "DEBUG: NRMLPModel constructor start" << std::endl;
		auto linear1 = torch::nn::Linear(InputSize, HiddenSize);
		std::cout << "DEBUG: Linear1 created" << std::endl;
		auto relu1 = torch::nn::ReLU();
		std::cout << "DEBUG: ReLU1 created" << std::endl;
		auto linear2 = torch::nn::Linear(HiddenSize, HiddenSize);
		std::cout << "DEBUG: Linear2 created" << std::endl;
		auto relu2 = torch::nn::ReLU();
		std::cout << "DEBUG: ReLU2 created" << std::endl;
		auto linear3 = torch::nn::Linear(HiddenSize, OutputSize);
		std::cout << "DEBUG: Linear3 created" << std::endl;

		Network = torch::nn::Sequential(linear1, relu1, linear2, relu2, linear3);
		std::cout << "DEBUG: NRMLPModel Network defined" << std::endl;
		register_module("Network", Network);
		std::cout << "DEBUG: NRMLPModel register_module done" << std::endl;
		std::cout << std::flush;
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
	std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

	std::cout << "=== QUICK TRAINING (TRAINEE TEST) ===" << std::endl;

	NR::NRModelProfile MyBotRig;
	
	std::string DataAssetPath = "Tests/Datasets/Foot_IK.json";
	if (!std::filesystem::exists(DataAssetPath))
	{
		DataAssetPath = "../Tests/Datasets/Foot_IK.json";
	}

	if (!NR::NRParse::LoadProfileFromJson(DataAssetPath, MyBotRig))
	{
		std::cerr << "Failed to load data asset: " << DataAssetPath << std::endl;
		return 1;
	}

	std::cout << "Profile loaded: " << MyBotRig.ProfileName << std::endl;

	int32_t InSize = MyBotRig.GetRequiredInputSize() + MyBotRig.GetRequiredTargetsSize(); // REVERTED TO 57 FOR TESTING
	int32_t OutSize = MyBotRig.GetRequiredOutputSize();

	std::cout << "Creating model..." << std::endl;
	std::cout << "InputSize: " << InSize << " OutputSize: " << OutSize << std::endl;
	std::cout << "DEBUG: about to create NRMLPModel" << std::endl;
	// Inject new model in NRTrainee
	auto model = std::make_shared<NRMLPModel>(
	    InSize,
	    64,
	    OutSize);
	std::cout << "Model created." << std::endl;
	
	std::cout << "Creating trainee..." << std::endl;
	std::cout << "DEBUG: about to create NRTrainee" << std::endl;
	NR::NRTrainee<float> trainee(model, MyBotRig, 1e-5);
	std::cout << "Trainee created." << std::endl;

	std::cout << "Starting training loop..." << std::endl;
	std::cout << std::flush;
	float last_loss = 0.0f;
	try
	{
		int32_t InSizeReq = MyBotRig.GetRequiredInputSize();
		int32_t TarSizeReq = MyBotRig.GetRequiredTargetsSize();
		int32_t TotalSize = InSizeReq + TarSizeReq;
		
		std::cout << "TotalSize: " << TotalSize << std::endl;
		std::vector<float> trainingData(TotalSize, 0.0f);

		// Pelvis pos WS — pelvis a 90cm de altura (Z up no UE5)
		trainingData[0] = 0.0f;   // X
		trainingData[1] = 0.0f;   // Y
		trainingData[2] = 90.0f;  // Z (altura)

		// Pelvis quat — identity
		trainingData[3] = 0.0f;  // X
		trainingData[4] = 0.0f;  // Y
		trainingData[5] = 0.0f;  // Z
		trainingData[6] = 1.0f;  // W

		// Thigh R offset LS (deslocamento lateral do hip)
		trainingData[7] = 10.0f;  // X (para a direita)
		trainingData[8] = 0.0f;   // Y
		trainingData[9] = 0.0f;   // Z

		// Thigh R quat LS — identity
		trainingData[13] = 1.0f;  // W (offset 10+3)

		// Calf R quat LS — identity
		trainingData[20] = 1.0f;  // W (offset 17+3)

		// Foot R quat LS — identity
		trainingData[27] = 1.0f;  // W (offset 24+3)

		// Thigh L offset LS
		trainingData[28] = -10.0f; // X (para a esquerda)
		trainingData[29] = 0.0f;
		trainingData[30] = 0.0f;

		// Thigh L quat LS — identity
		trainingData[34] = 1.0f;  // W (offset 31+3)

		// Calf L quat LS — identity
		trainingData[41] = 1.0f;  // W (offset 38+3)

		// Foot L quat LS — identity
		trainingData[48] = 1.0f;  // W (offset 45+3)

		// Ground normal R (apontando pra cima)
		trainingData[49] = 0.0f;
		trainingData[50] = 0.0f;
		trainingData[51] = 1.0f;

		// Has hit R
		trainingData[52] = 1.0f;

		// Ground normal L
		trainingData[53] = 0.0f;
		trainingData[54] = 0.0f;
		trainingData[55] = 1.0f;

		// Has hit L
		trainingData[56] = 1.0f;

		// Foot R target PS — pé deve ficar ~80cm abaixo do pelvis
		// Hip está em (10, 0, 90) WS. Target no chão: (10, 0, 5)
		// Em PS (relativo ao pelvis): (10, 0, 5) - (0,0,90) = (10, 0, -85)
		// MAS alcance máximo = L1+L2 = 45+40 = 85. Vamos usar 80 pra ficar dentro:
		trainingData[57] = 10.0f;  // X
		trainingData[58] = 0.0f;   // Y
		trainingData[59] = -20.0f; // Z (80cm abaixo do pelvis — dentro do alcance do hip)

		// Foot L target PS
		trainingData[60] = -10.0f;
		trainingData[61] = 0.0f;
		trainingData[62] = 10.0f;

		std::cout << "Data prepared. Starting loop." << std::endl;
		for (int i = 0; i < 10000; ++i)
		{
			float loss = trainee.TrainStep(trainingData, 45.75f, 41.70f, 45.75f, 41.70f);
			if (loss < 0.0001f)
			{
				std::cout << "Convergence reached at epoch " << i << std::endl;
				break;
			}
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "EXCEPTION: " << e.what() << std::endl;
		return 1;
	}
	std::cout << std::flush;

	std::cout << "Final Loss: " << last_loss << std::endl;

	if (last_loss < 0.0001f)
	{
		std::cout << "TEST PASSED: Loss decreased significantly." << std::endl;
	}
	else
	{
		std::cout << "TEST FAILED: Loss did not decrease as expected." << std::endl;
	}

	return 0;
}
