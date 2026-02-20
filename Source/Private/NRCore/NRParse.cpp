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

				// Popula Variables (se existirem)
				if (schema.contains("Variables"))
				{
					for (const auto& item : schema["Variables"])
					{
						NRDataBlock block;
						block.Name = item.value("Name", "Unknown_Var");
						block.FloatCount = item.value("Size", 1);
						block.bIsTarget = item.value("IsTarget", false);
						OutProfile.Variables.push_back(block);
					}
				}

				// Popula Bones (se existirem)
				if (schema.contains("Bones"))
				{
					for (const auto& item : schema["Bones"])
					{
						NRDataBlock block;
						block.Name = item.value("Name", "Unknown_Bone");
						block.FloatCount = item.value("Size", 3);
						block.bIsTarget = item.value("IsTarget", false);
						OutProfile.Bones.push_back(block);
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
