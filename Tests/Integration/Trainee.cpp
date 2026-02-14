#include "NRInterfaces/INRModel.h"
#include "NRTrainee/NRTrainee.h"
#include <iostream>

// Implementação simples de INRModel para teste
class MockModel : public NR::INRModel<float>
{
public:
	MockModel()
	{
		linear = register_module("linear", torch::nn::Linear(3, 3));
	}

	torch::Tensor Forward(torch::Tensor Input) override
	{
		return linear->forward(Input);
	}

	void SaveModel(const std::string& /*FilePath*/) override {}
	void LoadModel(const std::string& /*FilePath*/) override {}

private:
	torch::nn::Linear linear{nullptr};
};

int main()
{
	std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

	std::cout << "=== QUICK TRAINING (TRAINEE TEST) ===" << std::endl;

	// Inject new model in NRTrainee
	auto model = std::make_shared<MockModel>();
	NR::NRTrainee<float> trainee(model, 0.1);

	std::vector<NR::NRVector3D> inputs = {
	    {0.0f, 0.0f, 0.0f},
	    {1.0f, 1.0f, 1.0f},
	    {2.0f, -1.0f, 0.5f}};

	std::vector<NR::NRVector3D> targets = {
	    {1.0f, 2.0f, 3.0f},
	    {2.0f, 3.0f, 4.0f},
	    {3.0f, 1.0f, 3.5f}};

	std::cout << "Starting training loop..." << std::endl;
	float last_loss = 0.0f;
	try
	{
		for (int i = 0; i < 100; ++i)
		{
			float loss = trainee.TrainStep(inputs, targets);
			if (i % 10 == 0)
			{
				std::cout << "Iteration " << i << " - Loss: " << loss << std::endl;
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
