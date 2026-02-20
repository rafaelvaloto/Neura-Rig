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

	int32_t InSize = MyBotRig.GetRequiredInputSize();
	int32_t OutSize = MyBotRig.GetRequiredOutputSize();

	std::cout << "Creating model..." << std::endl;
	std::cout << "InputSize: " << InSize << " OutputSize: " << OutSize << std::endl;
	// Inject new model in NRTrainee
	auto model = std::make_shared<NRMLPModel>(
	    InSize,
	    64,
	    OutSize);
	std::cout << "Model created." << std::endl;
	
	std::cout << "Creating trainee..." << std::endl;
	NR::NRTrainee<float> trainee(model, MyBotRig, 0.01);
	std::cout << "Trainee created." << std::endl;

	std::cout << "=== generating data ===" << std::endl;

	// Com a nova estrutura, os tamanhos de entrada/saída podem não ser múltiplos de 3.
	// Vamos gerar dados em blocos de 3 floats (NRVector3D) suficientes para cobrir todos os floats necessários.
	auto ceilDiv3 = [](int32_t v) { return (v + 2) / 3; };
	int32_t InputTripletsPerSample = ceilDiv3(InSize);
	int32_t TargetTripletsPerSample = ceilDiv3(OutSize);

	std::cout << "Triplets per sample -> Inputs: " << InputTripletsPerSample << ", Targets: " << TargetTripletsPerSample << std::endl;

	std::vector<NR::NRVector3D> inputs;
	inputs.reserve((InputTripletsPerSample) * 2);
	// Sample 1 (zeros)
	for(int i=0; i<InputTripletsPerSample; ++i) inputs.push_back({0.0f, 0.0f, 0.0f});
	// Sample 2 (uns)
	for(int i=0; i<InputTripletsPerSample; ++i) inputs.push_back({1.0f, 1.0f, 1.0f});

	std::vector<NR::NRVector3D> targets;
	targets.reserve((TargetTripletsPerSample) * 2);
	// Sample 1
	for(int i=0; i<TargetTripletsPerSample; ++i) targets.push_back({0.5f, 0.5f, 0.5f});
	// Sample 2
	for(int i=0; i<TargetTripletsPerSample; ++i) targets.push_back({1.5f, 1.5f, 1.5f});

	std::cout << "Starting training loop..." << std::endl;
	float last_loss = 0.0f;
	try
	{
		for (int i = 0; i < 1000; ++i) // Limite alto de segurança
		{
			float loss = trainee.TrainStep(inputs, targets);
			if (i % 10 == 0)
			{
				std::cout << "Iteration " << i << " - Loss: " << loss << std::endl;
			}

			// Se a Loss já for insignificante, para o treino
			if (loss < 0.0001f)
			{
				std::cout << "Convergence reached at epoch " << i << std::endl;
				last_loss = loss;
				break;
			}
			last_loss = loss;
		}
	}
	catch (const std::exception& e)
	{
		std::cerr << "EXCEPTION: " << e.what() << std::endl;
		return 1;
	}
	std::cout << std::flush;

	std::cout << "Final Loss: " << last_loss << std::endl;

	if (last_loss < 0.1f)
	{
		std::cout << "TEST PASSED: Loss decreased significantly." << std::endl;
	}
	else
	{
		std::cout << "TEST FAILED: Loss did not decrease as expected." << std::endl;
	}

	return 0;
}
