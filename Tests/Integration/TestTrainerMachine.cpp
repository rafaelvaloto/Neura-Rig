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

#include "Core/Parse.h"
#include "Network/NetworkServer.h"
#include "Network/NetworkClient.h"
#include "Solver/Solver.h"
#include "Trainee/Trainee.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#ifdef _WIN32
#include <Windows.h>
#endif


class TestQuatFromUnreal : public NR::IQuat
{
public:
	[[nodiscard]] NR::Quat ToQuat(float pitch, float yaw, float roll) const override
	{
		float cr = std::cos(roll  * 0.5f);
		float sr = std::sin(roll  * 0.5f);
		float cp = std::cos(pitch * 0.5f);
		float sp = std::sin(pitch * 0.5f);
		float cy = std::cos(yaw   * 0.5f);
		float sy = std::sin(yaw   * 0.5f);

		// q = q_Yaw * q_Pitch * q_Roll
		float x =  cy * cp * sr - sy * sp * cr;
		float y =  cy * sp * cr + sy * cp * sr;
		float z =  sy * cp * cr - cy * sp * sr;
		float w =  cy * cp * cr + sy * sp * sr;
		
		return torch::tensor({x, y, z, w}, torch::kFloat32);
	}

	NR::Vec3 ToEuler(const NR::Quat& q) const override
	{
		auto q_norm = torch::nn::functional::normalize(q, torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));

		auto x = q_norm.select(-1, 0);
		auto y = q_norm.select(-1, 1);
		auto z = q_norm.select(-1, 2);
		auto w = q_norm.select(-1, 3);

		// Roll (X)
		auto sinr_cosp = 2.0f * (w * x + y * z);
		auto cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
		auto roll = torch::atan2(sinr_cosp, cosr_cosp);

		// Pitch (Y)
		auto sinp = 2.0f * (w * y - z * x);
		auto pitch = torch::asin(torch::clamp(sinp, -0.999999f, 0.999999f));

		auto siny_cosp = 2.0f * (w * z + x * y);
		auto cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
		auto yaw = torch::atan2(siny_cosp, cosy_cosp);

		return torch::stack({pitch, yaw, roll}, -1);
	}
};

class NRMultiHeadModel : public NR::IModel<float>
{
public:
	torch::nn::Sequential backbone{nullptr};
	torch::nn::Linear head_leg_r{nullptr};
	torch::nn::Linear head_leg_l{nullptr};
	torch::nn::Linear head_pelvis_ik{nullptr};

	NRMultiHeadModel(int64_t in_size, int64_t hidden, int64_t /*out_size*/)
	{
		backbone = register_module("backbone", torch::nn::Sequential(
			                           torch::nn::Linear(in_size, hidden),
			                           torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden})),
			                           torch::nn::ELU(),
			                           torch::nn::Linear(hidden, hidden),
			                           torch::nn::ELU()
			                           ));

		head_pelvis_ik = register_module("head_pelvis_ik", torch::nn::Linear(hidden, 7));
		head_leg_r = register_module("head_leg_r", torch::nn::Linear(hidden, 21));
		head_leg_l = register_module("head_leg_l", torch::nn::Linear(hidden, 21));
	}

	torch::Tensor Forward(torch::Tensor x) override
	{
		auto feat = backbone->forward(x);
		auto h_pelvis = head_pelvis_ik->forward(feat);
		auto h_leg_r = head_leg_r->forward(feat);
		auto h_leg_l = head_leg_l->forward(feat);

		// Apply tanh to keep translations within reasonable bounds if needed,
		// but typically we want raw values for IK.
		// For orientations (Quaternions), we might want to normalize later.

		// Outputs: pelvis_ik (7), legs_ik_r (21), legs_ik_l (21)
		return torch::cat({h_pelvis, h_leg_r, h_leg_l}, 1);
	}

	void SaveModel(const std::string& FilePath) override
	{
		torch::save(shared_from_this(), FilePath);
	}

	void LoadModel(const std::string& FilePath) override
	{
		auto self = shared_from_this();
		torch::load(self, FilePath);
	}
};

bool saveModel = false;

int main()
{
	using namespace NR;

	NetworkServer Server;
	if (!Server.Start(8005))
	{
		std::cerr << "CRITICAL: Could not start main server on port 8005." << std::endl;
		return 1;
	}

	NRModelProfile ActiveProfile;
	Rules ActiveRules;
	std::shared_ptr<Solver> NRSolver = nullptr;
	std::shared_ptr<IQuat> CustomQuat = nullptr;

	// std::string DataAssetPath_IK = "Tests/Datasets/Foot_IK.json";
	std::string DataAssetPath_IK = "Tests/Datasets/Rest_Pose_IK.json";
	std::string DataAssetPath_SK = "Tests/Datasets/Foot_SK.json";
	std::string DataAssetPath_TW = "Tests/Datasets/Foot_TW.json";

	if (!std::filesystem::exists(DataAssetPath_IK))
	{
		//DataAssetPath_IK = "../Tests/Datasets/Foot_IK.json";
		DataAssetPath_IK = "../Tests/Datasets/Rest_Pose_IK.json";
		DataAssetPath_SK = "../Tests/Datasets/Foot_SK.json";
		DataAssetPath_TW = "../Tests/Datasets/Foot_TW.json";
	}

	DataAssetPath_IK = std::filesystem::absolute(DataAssetPath_IK).string();
	DataAssetPath_SK = std::filesystem::absolute(DataAssetPath_SK).string();
	DataAssetPath_TW = std::filesystem::absolute(DataAssetPath_TW).string();

	if (!Parse::LoadIKFromJson(DataAssetPath_IK, ActiveProfile))
	{
		std::cerr << "Failed to load IK asset: " << DataAssetPath_IK << std::endl;
		return 1;
	}

	if (!CustomQuat)
	{
		CustomQuat = std::make_shared<TestQuatFromUnreal>();
	}

	if (!Parse::LoadSKFromJson(DataAssetPath_SK, ActiveProfile.Skeleton, CustomQuat.get()))
	{
		std::cerr << "Failed to load SK asset: " << DataAssetPath_SK << std::endl;
		return 1;
	}

	if (!Parse::LoadTWFromJson(DataAssetPath_TW, ActiveProfile.TrainingWeights))
	{
		std::cerr << "Failed to load TW asset: " << DataAssetPath_TW << std::endl;
		return 1;
	}

	std::cout << "----------------------------------" << std::endl;
	std::cout << "Profile loaded: " << ActiveProfile.ProfileName << std::endl;
	std::cout << " -> Input Size: " << ActiveProfile.GetRequiredInputSize() << std::endl;
	std::cout << " -> Output Size: " << ActiveProfile.GetRequiredOutputSize() << std::endl;

	auto InputSize = ActiveProfile.GetRequiredInputSize();
	auto OutSize = ActiveProfile.GetRequiredOutputSize();
	auto Model = std::make_shared<NRMultiHeadModel>(InputSize, 512, OutSize);
	std::cout << "Model created!" << std::endl;

	auto NRTrainee = std::make_shared<Trainee<float> >(Model, CustomQuat.get(), ActiveProfile, ActiveRules, 4e-3);
	std::cout << "Model trainee configuration!" << std::endl;

	std::string ModelSavePath = "Datasets/trained_model.pt";
	if (!std::filesystem::exists(ModelSavePath))
	{
		ModelSavePath = "../Tests/Datasets/trained_model.pt"; // Fallback para carregar da raiz se necessário, ou outros setups
	}
	
	// Garantir que temos o caminho absoluto
	ModelSavePath = std::filesystem::absolute(ModelSavePath).string();
	
	if (std::filesystem::exists(ModelSavePath))
	{
		try
		{
			Model->LoadModel(ModelSavePath);
			std::cout << ">>> Modelo carregado com sucesso de: " << ModelSavePath << std::endl;
		}
		catch (const std::exception& e)
		{
			std::cerr << ">>> Erro ao carregar modelo: " << e.what() << ". Iniciando do zero." << std::endl;
		}
	}

	if (Server.IsRunning())
	{
		NetworkClient ClientDebug;
		NetworkClient ClientSolver;
		
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

				if (NRTrainee)
				{
					int32_t requiredSize = ActiveProfile.GetRequiredInputSize();
					if (data.size() < static_cast<size_t>(requiredSize))
					{
						std::cerr << "[Server] Incomplete data received: " << data.size() << " floats, expected at least " << requiredSize << std::endl;
						continue;
					}

					float loss = NRTrainee->TrainStep(data);
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
							Model->SaveModel(ModelSavePath);
							std::cout << "[Checkpoint] Modelo salvo automaticamente em: " << ModelSavePath << " (Frame: " << frameCounter << ")" << std::endl;
						}
						catch (const std::exception& e)
						{
							std::cerr << "[Erro] Falha ao salvar checkpoint: " << e.what() << std::endl;
						}
					}

					if (NRTrainee->IdealTargets.defined())
					{
						const float* dDataPtr = NRTrainee->IdealTargets.data_ptr<float>();
						auto dNumElements = NRTrainee->IdealTargets.numel();
						std::vector<float> debugData(dDataPtr, dDataPtr + dNumElements);

						ClientDebug.Send(debugData, "127.0.0.1", 8007);
					}

					if (!NRSolver && loss < 2.0f)
					{
						NRSolver = std::make_shared<Solver>(Model, ActiveProfile);
						std::cout << "=== SWITCHING TO SOLVER MODE ===" << std::endl;
					}

					if (NRSolver)
					{
						std::vector<float> solveInput(InputSize);
						std::memcpy(solveInput.data(), data.data(), InputSize * sizeof(float));

						std::vector<float> predicted = NRSolver->Solve(solveInput);
						ClientSolver.Send(predicted, "127.0.0.1", 8006);
					}
				}
			}
		}
	}
	return 0;
}
