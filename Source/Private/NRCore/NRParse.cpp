// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"
#include <fstream>
#include <iostream>

#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)
#pragma warning(disable : 4702)
#pragma warning(disable : 4100)
#include <nlohmann/json.hpp>
#pragma warning(pop)

using json = nlohmann::json;

namespace NR
{
	bool NRParse::LoadProfileFromJson(const std::string& FilePath, NRModelProfile& OutProfile)
	{
		std::ifstream file(FilePath);
		if (!file.is_open())
		{
			std::cerr << "[NeuraRig] Erro: Nao foi possivel abrir o arquivo: " << FilePath << std::endl;
			return false;
		}

		try
		{
			// Faz o parse do ficheiro para um objecto JSON
			json j;
			file >> j;

			// Extrai o nome do profile
			OutProfile.ProfileName = j.value("Profile", "Unknown_Profile");

			if (j.contains("Schema"))
			{
				auto schema = j["Schema"];

				// Popula Inputs
				if (schema.contains("Inputs"))
				{
					for (const auto& item : schema["Inputs"])
					{
						NRDataBlock block;
						block.Name = item.value("Name", "Unknown_Input");
						block.FloatCount = item.value("Size", 1);
						block.bIsTarget = false;
						OutProfile.Inputs.push_back(block);
					}
				}

				// Popula Outputs
				if (schema.contains("Outputs"))
				{
					for (const auto& item : schema["Outputs"])
					{
						NRDataBlock block;
						block.Name = item.value("Name", "Unknown_Output");
						block.FloatCount = item.value("Size", 1);
						block.bIsTarget = true;
						OutProfile.Outputs.push_back(block);
					}
				}
			}
			return true;
		}
		catch (const json::exception& e)
		{
			std::cerr << "[NeuraRig] Erro de formatacao no JSON: " << e.what() << std::endl;
			return false;
		}
	}

} // namespace NR
