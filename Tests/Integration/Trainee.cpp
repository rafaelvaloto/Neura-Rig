// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRInterfaces/INRModel.h"
#include "NRTrainee/NRTrainee.h"
#include <iostream>

// Implementação simples de INRModel para teste
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

	NR::NRRigDescription MyBotRig = {3, 1};

	// Inject new model in NRTrainee
	auto model = std::make_shared<NRMLPModel>(
	    MyBotRig.GetRequiredInputSize(),
	    64,
	    MyBotRig.GetRequiredOutputSize());
	NR::NRTrainee<float> trainee(model, MyBotRig, 0.1);

	std::cout << "=== model ===" << std::endl;
	// 2 amostras de treino
	std::vector<NR::NRVector3D> inputs = {
	    {0.0f, 0.0f, 0.0f}, // Alvo amostra 1
	    {1.0f, 1.0f, 1.0f}  // Alvo amostra 2
	};

	std::cout << "=== inputs ===" << std::endl;
	// Para cada entrada, precisamos de 3 saídas (uma para cada osso)
	std::vector<NR::NRVector3D> targets = {
	    // Saída para Amostra 1 (3 ossos)
	    {0.1f, 0.1f, 0.1f},
	    {0.2f, 0.2f, 0.2f},
	    {0.3f, 0.3f, 0.3f},

	    // Saída para Amostra 2 (3 ossos)
	    {1.1f, 1.1f, 1.1f},
	    {1.2f, 1.2f, 1.2f},
	    {1.3f, 1.3f, 1.3f}};

	std::cout << "=== targets ===" << std::endl;
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
