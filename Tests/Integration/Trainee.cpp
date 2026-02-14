#include <iostream>
#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)
#pragma warning(disable : 4702)
#include <torch/torch.h>
#pragma warning(pop)

// Helper function to translate the AI output ID into readable text
std::string ActionToText(int64_t id)
{ // Alterado para int64_t
	if (id == 0)
	{
		return "RUN AWAY (Low Health)";
	}
	if (id == 1)
	{
		return "ATTACK   (Close + Good Health)";
	}
	return "HEAL     (Far/Safe)";
}

int main()
{
	std::cout << "PyTorch version from parts: "
	          << TORCH_VERSION_MAJOR << "."
	          << TORCH_VERSION_MINOR << "."
	          << TORCH_VERSION_PATCH << std::endl;
	std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

	std::cout << "=== QUICK TRAINING (NO SAVE) ===" << std::endl;

	// Inputs: [Distance, Health] (0.0 to 1.0)
	int32_t n = 1000;
	torch::Tensor inputs = torch::rand({n, 2});
	torch::Tensor targets = torch::zeros({n}, torch::kLong);

	float* data_in = inputs.data_ptr<float>();
	long long* data_tgt = targets.data_ptr<long long>();

	for (int32_t i = 0; i < n; i++)
	{
		float dist = data_in[i * 2 + 0];
		float health = data_in[i * 2 + 1];

		if (health < 0.3)
		{
			data_tgt[i] = 0;
		}
		else if (dist < 0.5)
		{
			data_tgt[i] = 1;
		}
		else
		{
			data_tgt[i] = 2;
		}
	}

	// --- 2. CREATE MODEL ---
	torch::nn::Sequential model(
	    torch::nn::Linear(2, 16), // 2 Inputs -> 16 Hidden Neurons
	    torch::nn::ReLU(),        // Activation
	    torch::nn::Linear(16, 3)  // 16 Neurons -> 3 Outputs (Actions)
	);

	auto optimizer = torch::optim::Adam(model->parameters(), 0.01);
	auto criterion = torch::nn::CrossEntropyLoss();

	// --- 3. TRAIN ---
	std::cout << "Training...";
	for (int32_t epoch = 1; epoch <= 500; ++epoch)
	{
		optimizer.zero_grad();
		auto output = model->forward(inputs);
		auto loss = criterion(output, targets);
		loss.backward();
		optimizer.step();

		if (epoch % 100 == 0)
		{
			std::cout << "."; // Minimalist progress bar
		}
	}
	std::cout << " Done!" << std::endl;

	// --- 4. VALIDATION TEST (CHECK IF IT WORKS) ---
	std::cout << "=== VALIDATION TEST ===" << std::endl;

	// Manual Test Cases
	model->eval(); // Set to evaluation mode

	for (int32_t i = 0; i < 3; i++)
	{
		float tests[3][2] = {
		    {0.1f, 0.1f}, // Case 1: Close and Dying -> Expected: RUN AWAY
		    {0.2f, 0.9f}, // Case 2: Close and Healthy -> Expected: ATTACK
		    {0.9f, 0.8f}  // Case 3: Far and Healthy -> Expected: HEAL
		};
		// Create tensor for a single case
		auto t_in = torch::tensor({{tests[i][0], tests[i][1]}});

		// Ask the AI
		auto t_out = model->forward(t_in);

		// Get the highest probability (argmax)
		int64_t decision = t_out.argmax(1).item<int64_t>();

		std::cout << "Scenario " << i + 1
		          << " [Dist: " << tests[i][0] << ", Health: " << tests[i][1] << "] "
		          << "-> AI decided: " << ActionToText(decision) << std::endl;
	}

	return 0;
}
