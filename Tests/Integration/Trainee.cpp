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
		auto linear1 = torch::nn::Linear(InputSize, HiddenSize);
		auto relu1 = torch::nn::ReLU();
		auto linear2 = torch::nn::Linear(HiddenSize, HiddenSize);
		auto relu2 = torch::nn::ReLU();
		auto linear3 = torch::nn::Linear(HiddenSize, OutputSize);

		Network = torch::nn::Sequential(linear1, relu1, linear2, relu2, linear3);
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

	try
	{
		int32_t InSizeReq = MyBotRig.GetRequiredInputSize();
		int32_t TarSizeReq = MyBotRig.GetRequiredTargetsSize();
		int32_t TotalSize = InSizeReq + TarSizeReq;
		
		std::cout << "TotalSize: " << TotalSize << std::endl;
		std::vector<float> trainingData(TotalSize, 0.0f);

		// Thigh R Pos (PS) - Offset 0
		trainingData[0] = 15.0f;  // X (um pouco para a direita do centro)

		// Thigh L Pos (PS) - Offset 3
		trainingData[3] = -15.0f; // X (um pouco para a esquerda do centro)

		// Bone Lengths - Offsets [6..9]
		float L1 = 45.75f;
		float L2 = 41.70f;
		trainingData[6] = L1; // L1 R
		trainingData[7] = L2; // L2 R
		trainingData[8] = L1; // L1 L
		trainingData[9] = L2; // L2 L

		// Ground Normals (Z up) - Offsets 10 e 14
		trainingData[12] = 1.0f; // Normal R (Z=1)
		trainingData[13] = 1.0f; // Has Hit R (1.0 = true)
		trainingData[16] = 1.0f; // Normal L (Z=1)
		trainingData[17] = 1.0f; // Has Hit L (1.0 = true)

		// Velocity e Pole Vector (Apontando para frente no Y) - Offsets 18 e 21
		trainingData[19] = 0.5f; // Velocity Y
		trainingData[21] = 1.0f; // Pole Vector Y (Joelho aponta para frente)

		// Bone Axis (O que veio do log do Unreal) - Offsets 24 e 27
		trainingData[24] = 1.0f;  // Axis R (X=1)
		trainingData[27] = -1.0f; // Axis L (X=-1)

		// Pelvis Quat (Identidade) - Offset 30
		trainingData[33] = 1.0f; // W=1

		// --- TARGETS ---

		// Foot Target R (PS) - Offset 34
		// Posição no chão: mesma vertical do quadril, abaixo a uma distância L1+L2
		trainingData[34] = 15.0f;
		trainingData[35] = 0.0f;
		trainingData[36] = -(L1 + L2 - 5.0f); // Z (Perna quase esticada)

		// Foot Target L (PS) - Offset 37
		trainingData[37] = -15.0f;
		trainingData[38] = 0.0f;
		trainingData[39] = -(L1 + L2 - 5.0f);

		float last_loss = 0.0f;
		std::cout << "Data prepared. Starting loop." << std::endl;
		for (int i = 0; i < 10000; ++i)
		{
			float loss = trainee.TrainStep(trainingData);
			std::cout << "==============================" << std::endl;
			std::cout << "Convergence loss " << last_loss << std::endl;
			std::cout << "==============================" << std::endl;
			last_loss = loss;
			if (loss <= 0.0025f)
			{
				std::cout << "Convergence reached at epoch "  << i << std::endl;
				break;
			}
		}

		std::cout << std::flush;

		std::cout << "Final Loss: " << last_loss << std::endl;
		if (last_loss <= 0.0025f)
		{
			std::cout << "TEST PASSED: Loss decreased significantly." << std::endl;
		}
		else
		{
			std::cout << "TEST FAILED: Loss did not decrease as expected." << std::endl;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "EXCEPTION: " << e.what() << std::endl;
		return 1;
	}

	return 0;
}
