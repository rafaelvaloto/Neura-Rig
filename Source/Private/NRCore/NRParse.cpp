// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.

#include "NRCore/NRParse.h"
#include <fstream>
#include <iostream>

#ifdef MUP_STRING_TYPE
#define MUP_STRING_TYPE std::string
#endif

#pragma warning(push)
#pragma warning(disable : 4244)
#pragma warning(disable : 4267)
#pragma warning(disable : 4996)
#pragma warning(disable : 4702)
#pragma warning(disable : 4100)
#include <nlohmann/json.hpp>
#pragma warning(pop)


#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Desativa o aviso de conversão de dados
#endif

#include "muParser.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using json = nlohmann::json;
using namespace mu;

namespace NR
{
	bool NRParse::LoadProfileFromJson(const std::string& FilePath, NRModelProfile& OutProfile)
	{
		std::ifstream file(FilePath);

		if (!file.is_open())
			return false;

		try
		{
			json j;
			file >> j;
			OutProfile.ProfileName = j.value("Profile", "Unknown");

			if (j.contains("Schema"))
			{
				auto schema = j["Schema"];

				// --- INPUTS & OUTPUTS (Já existentes) ---
				for (const auto& item : schema["Inputs"])
				{
					OutProfile.Inputs.push_back({item["Name"], item["Offset"], item["Size"]});
				}
				for (const auto& item : schema["Outputs"])
				{
					OutProfile.Outputs.push_back({item["Name"], item["Offset"], item["Size"]});
				}

				// --- RULES ---
				if (schema.contains("Rules"))
				{
					for (const auto& r : schema["Rules"])
					{
						NRRule rule;
						rule.Name = r["Name"];

						// Constantes
						for (auto& el : r["Constants"].items())
						{
							rule.Constants[el.key()] = el.value().is_number() ? el.value().get<double>() : 0.0;
						}

						// Variáveis (Mapeamento de nomes)
						for (auto& el : r["Variables"].items())
						{
							rule.Variables[el.key()] = el.value().get<std::vector<std::string> >();
						}

						// Lógica base
						for (auto& el : r["Logic"].items())
						{
							rule.Logic[el.key()] = el.value();
						}

						// Fases
						for (const auto& p : r["Phases"])
						{
							NRRule::Phase phase;
							phase.Id = p["id"];
							phase.Condition = p["condition"];
							for (auto& el : p.items())
							{
								if (el.key() != "id" && el.key() != "condition")
								{
									phase.Formulas[el.key()] = el.value();
								}
							}
							rule.Phases.push_back(phase);
						}
						OutProfile.Rules.push_back(rule);
					}
				}

				if (schema.contains("Bindings"))
				{
					for (const auto& b : schema["Bindings"])
					{
						NRBinding binding;
						binding.BoneName = b["Name"];
						binding.RuleName = b["Target"];
						OutProfile.Bindings.push_back(binding);
					}
				}
			}
			return true;
		}
		catch (std::exception& e)
		{
			std::cerr << "Erro no JSON: " << e.what() << std::endl;
			return false;
		}
	}

} // namespace NR
