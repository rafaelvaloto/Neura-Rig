// Project: NeuralRig
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
#pragma warning(disable : 4244) // Deactivates the data conversion warning
#endif

#include "muParser.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using json = nlohmann::ordered_json;
using namespace mu;

namespace NR
{
	bool Parse::LoadProfileFromJson(const std::string& FilePath, NRModelProfile& OutProfile)
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

				// --- INPUTS & OUTPUTS ---
				if (schema.contains("Inputs"))
				{
					for (const auto& item : schema["Inputs"])
					{
						OutProfile.Inputs.push_back({item["Name"], item["Offset"], item["Size"]});
					}
				}

				if (schema.contains("Outputs"))
				{
					for (const auto& item : schema["Outputs"])
					{
						OutProfile.Outputs.push_back({item["Name"], item["Offset"], item["Size"]});
					}
				}

				// --- BINDINGS & RULES ---
				if (schema.contains("Bindings"))
				{
					for (const auto& b : schema["Bindings"])
					{
						NRBinding binding;
						binding.Size = b["Size"];
						binding.Offset = b["Offset"];
						binding.BoneName = b["Name"];
						binding.RuleName = b["Target"];

						if (schema.contains("Rules"))
						{
							auto& rules = schema["Rules"];
							auto it = std::find_if(rules.begin(), rules.end(), [&](const auto& rule) {
								return rule["Name"] == b["Target"];
							});

							if (it != rules.end())
							{
								NRRule rule;
								rule.Name = it->at("Name");

								// Parameters (Constants)
								if (schema.contains("Parameters") && schema["Parameters"].contains("Constants"))
								{
									for (auto& el : schema["Parameters"]["Constants"].items())
									{
										rule.Constants[el.key()] = el.value().is_number() ? el.value().get<double>() : 0.0;
									}
								}

								// Parameters (Variables)
								if (schema.contains("Parameters") && schema["Parameters"].contains("Variables"))
								{
									for (auto& el : schema["Parameters"]["Variables"].items())
									{
										NRVars vars;
										vars.Name = el.key();
										vars.List = el.value().get<std::vector<std::string>>();
										rule.Variables.push_back(vars);
									}
								}

								// Logic
								if (it->contains("Logic"))
								{
									for (auto& el : it->at("Logic").items())
									{
										NRLogic logic;
										logic.Name = el.key();
										logic.Expr = el.value();
										rule.Logic.push_back(logic);
										std::cout << "Logic: " << logic.Name << " : " << logic.Expr << std::endl;
									}
								}

								// Rotation Limits
								if (it->contains("Limits"))
								{
									auto& lim = it->at("Limits");
									rule.Limits.MinX = lim.value("MinX", -360.0f);
									rule.Limits.MaxX = lim.value("MaxX", 360.0f);
									rule.Limits.MinY = lim.value("MinY", -360.0f);
									rule.Limits.MaxY = lim.value("MaxY", 360.0f);
									rule.Limits.MinZ = lim.value("MinZ", -360.0f);
									rule.Limits.MaxZ = lim.value("MaxZ", 360.0f);
								}

								if (it->contains("Phases"))
								{
									for (auto& phase_item : it->at("Phases"))
									{
										NRRule::Phase phase;
										phase.Id = phase_item.at("id");
										phase.Condition = phase_item.at("condition");

										for (auto& el2 : phase_item.items())
										{
											if (el2.key() != "condition" && el2.key() != "id")
											{
												NRFormula formula;
												formula.Name = el2.key();
												formula.Expr = el2.value().get<std::string>();
												phase.Formulas.push_back(formula);
												std::cout << "Phase Formula (" << phase.Id << "): " << formula.Name << " : " << formula.Expr << std::endl;
											}
										}
										rule.Phases.push_back(phase);
									}
								}
								binding.Rules.push_back(rule);
							}
						}
						OutProfile.Bindings.push_back(binding);
					}
				}
			}
			return true;
		}
		catch (std::exception& e)
		{
			std::cerr << "JSON Error: " << e.what() << std::endl;
			return false;
		}
	}

} // namespace NR
