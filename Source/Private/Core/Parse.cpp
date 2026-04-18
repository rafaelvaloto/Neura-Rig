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
		return LoadIKFromJson(FilePath, OutProfile);
	}

	bool Parse::LoadIKFromJson(const std::string& FilePath, NRModelProfile& OutProfile)
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
			std::cerr << "JSON IK Error: " << e.what() << std::endl;
			return false;
		}
	}

	bool Parse::LoadSKFromJson(const std::string& FilePath, NRSkeleton& OutSkeleton)
	{
		std::ifstream file(FilePath);
		if (!file.is_open()) return false;

		try
		{
			json j;
			file >> j;
			if (!j.contains("Schema")) return false;
			auto schema = j["Schema"];

			auto parseBone = [](const json& b) {
				NRSkeleton::Bone bone;
				bone.Name = b.value("Name", "");
				bone.Size = b.value("Size", 0);
				bone.Offset = b.value("Offset", 0);
				auto getFloat = [](const json& p, const std::string& key) {
					auto val = p.value(key, "0.0");
					std::replace(val.begin(), val.end(), ',', '.');
					try {
						return std::stof(val);
					} catch (...) {
						return 0.0f;
					}
				};
				if (b.contains("Pose")) {
					auto& p = b["Pose"];
					bone.RestPose.x = getFloat(p, "x") * 0.01;
					bone.RestPose.y = getFloat(p, "y") * 0.01;
					bone.RestPose.z = getFloat(p, "z") * 0.01;

					if (p.contains("Pitch") || p.contains("Yaw") || p.contains("Roll")) {
						float pitch = DegToRad(getFloat(p, "Pitch"));
						float yaw   = DegToRad(getFloat(p, "Yaw"));
						float roll  = DegToRad(getFloat(p, "Roll"));
						Quat q = Quat::FromUnrealRotator(pitch, yaw, roll);
						bone.RestPose.q1 = q.x;
						bone.RestPose.q2 = q.y;
						bone.RestPose.q3 = q.z;
						bone.RestPose.qw = q.w;

						std::cout << "Quat: " << pitch << ", " << yaw << ", " << roll << std::endl;
						std::cout << "Quat: " << q.x << ", " << q.y << ", " << q.z << ", " << q.w << std::endl;
					} else {
						bone.RestPose.q1 = getFloat(p, "q1");
						bone.RestPose.q2 = getFloat(p, "q2");
						bone.RestPose.q3 = getFloat(p, "q3");
						bone.RestPose.qw = getFloat(p, "qw");
					}
				}
				if (b.contains("Limits")) {
					auto& lim = b["Limits"];
					bone.Limits.MinX = lim.value("MinX", -360.0f);
					bone.Limits.MaxX = lim.value("MaxX", 360.0f);
					bone.Limits.MinY = lim.value("MinY", -360.0f);
					bone.Limits.MaxY = lim.value("MaxY", 360.0f);
					bone.Limits.MinZ = lim.value("MinZ", -360.0f);
					bone.Limits.MaxZ = lim.value("MaxZ", 360.0f);
				}
				if (b.contains("Childrens")) {
					bone.ChildrenIndices = b["Childrens"].get<std::vector<int32_t>>();
				}
				return bone;
			};

			if (schema.contains("Parent")) {
				OutSkeleton.Parent = parseBone(schema["Parent"]);
			}

			if (schema.contains("Rest")) {
				for (const auto& chain : schema["Rest"]) {
					std::vector<NRSkeleton::Bone> boneChain;
					for (const auto& b : chain) {
						boneChain.push_back(parseBone(b));
					}
					OutSkeleton.Rest.push_back(boneChain);
				}
			}
			return true;
		}
		catch (std::exception& e) {
			std::cerr << "JSON SK Error: " << e.what() << std::endl;
			return false;
		}
	}

	bool Parse::LoadTWFromJson(const std::string& FilePath, NRTrainingWeights& OutWeights)
	{
		std::ifstream file(FilePath);
		if (!file.is_open()) return false;

		try
		{
			json j;
			file >> j;
			if (!j.contains("Schema")) return false;
			auto schema = j["Schema"];

			if (schema.contains("HyperParameters")) {
				auto& hp = schema["HyperParameters"];
				OutWeights.HyperParameters.LearningRate = hp.value("LearningRate", 0.0001f);
				OutWeights.HyperParameters.EmaAlpha = hp.value("EmaAlpha", 0.15f);
				OutWeights.HyperParameters.MaxCandidates = hp.value("MaxCandidates", 5);
			}

			if (schema.contains("LossWeights")) {
				for (auto& el : schema["LossWeights"].items()) {
					NRWeight w;
					w.Weight = el.value().value("Weight", 1.0f);
					w.Description = el.value().value("Description", "");
					OutWeights.LossWeights[el.key()] = w;
				}
			}

			if (schema.contains("BoneSpecificBias")) {
				for (const auto& b : schema["BoneSpecificBias"]) {
					NRTrainingWeights::BoneBias bias;
					bias.Name = b.value("Name", "");
					bias.PositionMultiplier = b.value("PositionMultiplier", 1.0f);
					bias.RotationMultiplier = b.value("RotationMultiplier", 1.0f);
					OutWeights.BoneSpecificBias.push_back(bias);
				}
			}
			return true;
		}
		catch (std::exception& e) {
			std::cerr << "JSON TW Error: " << e.what() << std::endl;
			return false;
		}
	}

} // namespace NR
