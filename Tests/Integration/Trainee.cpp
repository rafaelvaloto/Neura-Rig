// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRInterfaces/INRModel.h"
#include "NRTrainee/NRTrainee.h"
#include "NRCore/NRParse.h"
#include <iostream>
#include <filesystem>

class NRMultiHeadModel : public NR::INRModel<float> {
public:
	torch::nn::Sequential backbone{nullptr};
	torch::nn::Linear head_r{nullptr}, head_l{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden) {
		backbone = register_module("backbone", torch::nn::Sequential(
			torch::nn::Linear(in_size, hidden),
			torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			torch::nn::ELU(),
			torch::nn::Linear(hidden, hidden),
			torch::nn::ELU()
		));

		head_r = register_module("head_r", torch::nn::Linear(hidden, 6));  // 1 Vec (foot_r)
		head_l = register_module("head_l", torch::nn::Linear(hidden, 6));  // 1 Vec (foot_l)
	}

	torch::Tensor Forward(torch::Tensor x) override {
		auto feat = backbone->forward(x);
		auto out_r = head_r->forward(feat);
		auto out_l = head_l->forward(feat);
		return torch::cat({out_r, out_l}, 1);
	}

	void SaveModel(const std::string& FilePath) override {
		torch::save(shared_from_this(), FilePath);
	}

	void LoadModel(const std::string& FilePath) override {
		torch::load(backbone, FilePath);
	}
};

int main()
{
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    std::cout << "=== NEURARIG DYNAMIC RULE TEST ===" << std::endl;

	using namespace NR;
    NRModelProfile MyBotRig;

	std::string DataAssetPath = "Tests/Datasets/Foot_IK.json";
	if (!std::filesystem::exists(DataAssetPath))
	{
		DataAssetPath = "../Tests/Datasets/Foot_IK.json"; // Fallback for some build configurations
	}

	if (!NRParse::LoadProfileFromJson(DataAssetPath, MyBotRig))
	{
		std::cerr << "Failed to load data asset: " << DataAssetPath << std::endl;
		return 1;
	}

    std::cout << "Profile loaded: " << MyBotRig.ProfileName << std::endl;

    // 2. Tamanhos automáticos baseados no JSON
    int32_t InSize = MyBotRig.GetRequiredInputSize();   // Deve ser 14
    int32_t OutSize = MyBotRig.GetRequiredOutputSize(); // Deve ser 6 (foot_r + foot_l)

    std::cout << "InputSize: " << InSize << " | OutputSize: " << OutSize << std::endl;

    // 3. Criar Modelo e Trainee
    auto model = std::make_shared<NRMultiHeadModel>(InSize, 512);
    NRRules Ev;
    auto trainee = std::make_shared<NR::NRTrainee<float>>(model, MyBotRig, Ev, 1e-4);

    try
    {
       // 4. Preparar apenas os INPUTS (14 floats)
       std::vector<float> trainingData(InSize, 0.0f);

       // Mapeamento conforme seu novo JSON:
       trainingData[0]  = 300.0f; // velocity

       // Component 1 (Right Leg)
       trainingData[1]  = 45.0f;  // bone_l1_cm1
       trainingData[2]  = 45.0f;  // bone_l2_cm1
       trainingData[3]  = 15.0f;  // ground_one X
       trainingData[4]  = 0.0f;   // ground_one Y
       trainingData[5]  = -90.0f; // ground_one Z
       trainingData[6]  = 1.0f;   // has_hit_one (true)

       // Component 2 (Left Leg)
       trainingData[7]  = 45.0f;  // bone_l1_cm2
       trainingData[8]  = 45.0f;  // bone_l2_cm2
       trainingData[9]  = -15.0f; // ground_two X
       trainingData[10] = 0.0f;   // ground_two Y
       trainingData[11] = -90.0f; // ground_two Z
       trainingData[12] = 1.0f;   // has_hit_two (true)

       // Time (delta_sec)
       trainingData[13] = 1.5f;   // No tempo 1.5s, o pé estará em fase de Swing (no ar)

       std::cout << "Data prepared. Targets will be calculated by NRRules during training." << std::endl;

       // 5. Loop de Convergência (Overfitting num único ponto)
       float last_loss = 0.0f;
       for (int i = 0; i < 5000; ++i)
       {
          // O TrainStep agora calcula o Reward internamente usando o muParser
          float loss = trainee->TrainStep(trainingData);

          if (i % 500 == 0) {
              std::cout << "Epoch " << i << " | Loss: " << loss << std::endl;
          }

          last_loss = loss;
          if (loss <= 0.0001f) {
             std::cout << "Convergence reached at epoch " << i << std::endl;
             break;
          }
       }

       std::cout << "Final Loss: " << last_loss << std::endl;

       // Validação
       if (last_loss <= 0.001f) {
          std::cout << "TEST PASSED: Model learned the procedural rule!" << std::endl;
       } else {
          std::cout << "TEST FAILED: Model could not converge to the rule." << std::endl;
       }
    }
    catch (const std::exception& e) {
       std::cerr << "EXCEPTION: " << e.what() << std::endl;
       return 1;
    }

    return 0;
}
