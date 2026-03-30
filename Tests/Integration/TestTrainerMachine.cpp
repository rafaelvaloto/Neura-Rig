// Project: NeuraRig
// Copyright (c) 2026 Rafael Valoto
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "NRCore/NRParse.h"
#include "NRNetwork/NRNetworkServer.h"
#include "NRNetwork/NRNetworkClient.h"
#include "NRSolver/NRSolver.h"
#include "NRTrainee/NRTrainee.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#include <Windows.h>
#endif

class NRMultiHeadModel : public NR::INRModel<float>
{
public:
	torch::nn::Sequential backbone{nullptr};
	torch::nn::Linear head_foot_r{nullptr};
	torch::nn::Linear head_foot_l{nullptr};
	torch::nn::Linear head_bal_r{nullptr};
	torch::nn::Linear head_bal_l{nullptr};
	torch::nn::Linear head_leg_r{nullptr};
	torch::nn::Linear head_leg_l{nullptr};
	torch::nn::Linear head_pelvis_ik{nullptr};
	torch::nn::Linear head_spine_ik{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden, int64_t /*out_size*/)
	{
		backbone = register_module("backbone", torch::nn::Sequential(
			                           torch::nn::Linear(in_size, hidden),
			                           torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			                           torch::nn::ELU(),
			                           torch::nn::Linear(hidden, hidden),
			                           torch::nn::ELU()
			                           ));

		head_foot_r = register_module("head_foot_r", torch::nn::Linear(hidden, 6));
		head_foot_l = register_module("head_foot_l", torch::nn::Linear(hidden, 6));
		head_bal_r = register_module("head_bal_r", torch::nn::Linear(hidden, 6));
		head_bal_l = register_module("head_bal_l", torch::nn::Linear(hidden, 6));
		head_leg_r = register_module("head_leg_r", torch::nn::Linear(hidden, 12));
		head_leg_l = register_module("head_leg_l", torch::nn::Linear(hidden, 12));
		head_pelvis_ik = register_module("head_pelvis_ik", torch::nn::Linear(hidden, 6));
		head_spine_ik = register_module("head_spine_ik", torch::nn::Linear(hidden, 6));
	}

	torch::Tensor Forward(torch::Tensor x) override
	{
		auto feat = backbone->forward(x);
		auto h_foot_l = head_foot_l->forward(feat);
		auto h_foot_r = head_foot_r->forward(feat);
		auto h_ball_l = head_bal_l->forward(feat);
		auto h_ball_r = head_bal_r->forward(feat);
		auto h_leg_l = head_leg_l->forward(feat);
		auto h_leg_r = head_leg_r->forward(feat);
		auto h_spine = head_spine_ik->forward(feat);
		auto h_pelvis = head_pelvis_ik->forward(feat);
		return torch::cat({h_foot_r, h_foot_l, h_ball_r, h_ball_l, h_leg_r, h_leg_l, h_pelvis, h_spine}, 1);
	}

	void SaveModel(const std::string& FilePath) override
	{
		torch::save(shared_from_this(), FilePath);
	}

	void LoadModel(const std::string& FilePath) override
	{
		torch::load(backbone, FilePath);
	}
};

bool saveModel = false;

int main()
{
	using namespace NR;

	NRNetworkServer Server;
	if (!Server.Start(8005))
	{
		std::cerr << "CRITICAL: Could not start main server on port 8005." << std::endl;
		return 1;
	}

	NRNetworkClient ClientSolver;
	NRNetworkClient ClientDebug;

	NRModelProfile activeProfile;
	NRRules activeRules;
	std::shared_ptr<NRSolver> solver = nullptr;

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
	std::cout << " -> Output Size: " << activeProfile.GetRequiredOutputSize() << std::endl;

	auto InputSize = activeProfile.GetRequiredInputSize();
	auto out_size = activeProfile.GetRequiredOutputSize();
	auto model = std::make_shared<NRMultiHeadModel>(InputSize, 512, out_size);
	std::cout << "Model created!" << std::endl;

	auto trainee = std::make_shared<NRTrainee<float> >(model, activeProfile, activeRules, 4e-3);
	std::cout << "Model trainee configuration!" << std::endl;

	// 1. Tentar carregar o modelo existente antes de começar
	const std::string ModelSavePath = "trained_model.pt";
	if (std::filesystem::exists(ModelSavePath))
	{
		try
		{
			model->LoadModel(ModelSavePath);
			std::cout << ">>> Modelo carregado com sucesso de: " << ModelSavePath << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cerr << ">>> Erro ao carregar modelo: " << e.what() << ". Iniciando do zero." << std::endl;
		}
	}

	if (Server.IsRunning())
	{
		std::cout << "----------------------------------" << std::endl;
		std::cout << "Server Started!" << std::endl;
		std::cout << "Main server listening on port: 8005" << std::endl;
		std::cout << "----------------------------------" << std::endl;
		std::cout << "Waiting for messages..." << std::endl;
		static int frameCounter = 0;
		while (true)
		{
			std::vector<float> data;
			if (Server.Receive(data))
			{
				if (data.empty())
				{
					continue;
				}

				if (trainee)
				{
					int32_t requiredSize = activeProfile.GetRequiredInputSize();
					if (data.size() < static_cast<size_t>(requiredSize))
					{
						std::cerr << "[Server] Incomplete data received: " << data.size() << " floats, expected at least " << requiredSize << std::endl;
						continue;
					}

					float loss = trainee->TrainStep(data);
					if (frameCounter++ % 30 == 0)
					{
						std::cout << "----------------------------------" << std::endl;
						std::cout << " Loss: " << loss << std::endl;
						std::cout << " frame counter:" << frameCounter << std::endl;
						std::cout << "----------------------------------" << std::endl;
					}

					// 2. Salvamento Periódico
					if (frameCounter % 500 == 0)
					{
						try
						{
							model->SaveModel(ModelSavePath);
							std::cout << "[Checkpoint] Modelo salvo automaticamente em: " << ModelSavePath << " (Frame: " << frameCounter << ")" << std::endl;
						}
						catch (const std::exception& e)
						{
							std::cerr << "[Erro] Falha ao salvar checkpoint: " << e.what() << std::endl;
						}
					}

					if (trainee->IdealTargets.defined())
					{
						const float* dDataPtr = trainee->IdealTargets.data_ptr<float>();
						auto dNumElements = trainee->IdealTargets.numel();
						std::vector<float> debugData(dDataPtr, dDataPtr + dNumElements);

						ClientDebug.Send(debugData, "127.0.0.1", 8007);
					}

					if (!solver && loss < 0.05)
					{
						solver = std::make_shared<NRSolver>(model, activeProfile);
						std::cout << "=== SWITCHING TO SOLVER MODE ===" << std::endl;
					}

					if (solver)
					{
						std::vector<float> solveInput(InputSize);
						std::memcpy(solveInput.data(), data.data(), InputSize * sizeof(float));

						std::vector<float> predicted = solver->Solve(solveInput);
						ClientSolver.Send(predicted, "127.0.0.1", 8006);
					}
				}
			}
		}
	}
	return 0;
}
