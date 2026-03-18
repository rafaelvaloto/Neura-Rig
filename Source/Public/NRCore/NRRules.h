// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include "muParser.h"
#include "NRTypes.h"

static mu::value_type fmod_wrapper(mu::value_type v1, mu::value_type v2)
{
	return std::fmod(v1, v2);
}

static mu::value_type clamp_wrapper(mu::value_type v, mu::value_type min, mu::value_type max)
{
	return std::clamp(v, min, max);
}

static mu::value_type pow_wrapper(mu::value_type v1, mu::value_type v2)
{
	return std::pow(v1, v2);
}

namespace NR
{
	class NRRules
	{
	public:
		double deltaTime = 0.5f;
		std::vector<mu::Parser> Parsers;
		std::vector<std::map<std::string, double>> Vars;

		NRRules() = default;

		void Setup(const NRRule& rule, int bindingIndex)
		{
			EnsureBinding(bindingIndex);

			std::cout << "Binding " << bindingIndex << "RuleName" << rule.Name << std::endl;
			for (auto const& [name, val] : rule.Constants)
			{
				DefineVariable(name, val, bindingIndex);
			}

			for (auto const& [varName, _inputList] : rule.Variables)
			{
				std::cout << "  [RULE " << bindingIndex << "] " << varName << std::endl;
				DefineVariable(varName, 0.0, bindingIndex);
			}

			for (auto const& [logicName, _expr] : rule.Logic)
			{
				std::cout << "  [RULE " << bindingIndex << "] " << logicName << std::endl;
				DefineVariable(logicName, 0.0, bindingIndex);
			}

			for (auto const& phase : rule.Phases)
			{
				DefineVariable(phase.Id + "_condition", 0.0, bindingIndex);
				for (auto const& [formulaName, _expr] : phase.Formulas)
				{
					if (formulaName != "Condition" && formulaName != "Id")
					{
						DefineVariable(phase.Id + "_" + formulaName, 0.0, bindingIndex);
					}
				}
			}
		}

		void SetTensorInputs(int bindingIndex, const NRRule& rule, const NRModelProfile& profile, const torch::Tensor& currentInput)
		{
			auto& varMap = Parsers[bindingIndex].GetVar();

			for (auto const& [varName, inputList] : rule.Variables)
			{
				auto pick = inputList[0];
				if (inputList.size() > bindingIndex)
				{
					pick = inputList[bindingIndex];
				}

				if (pick == "t_cycle")
				{
					deltaTime += profile.GetInputBoneValue(currentInput, pick).item<double>();
					if (auto it = varMap.find(varName); it != varMap.end())
					{
						auto& logics = rule.Logic;
						auto T_it = std::find_if(logics.begin(), logics.end(), [&](const auto& logic) {
							return logic.Name == "T_gait";
						});
						const auto T_gait = Eval(bindingIndex, T_it->Expr);
						if (T_gait > 0.0)
						{
							deltaTime = std::fmod(deltaTime, T_gait);
						}

						*(it->second) = deltaTime;
					}
					continue;
				}

				if (auto it = varMap.find(varName); it != varMap.end())
				{
					*(it->second) = profile.GetInputBoneValue(currentInput, pick).item<double>();
				}
			}
		}

		double Eval(int bindingIndex, const std::string& expression)
		{
			try
			{
				Parsers[bindingIndex].SetExpr(expression);

				// DEBUG: dump what the parser actually sees
				// auto& varStorage = Vars[bindingIndex];
				// for (auto& [name, val] : varStorage)
				// {
				// 	std::cout << "  [STORAGE " << bindingIndex << "] " << name
				// 			  << " val=" << val << std::endl;
				// }

				return Parsers[bindingIndex].Eval();
			}
			catch (mu::Parser::exception_type& e)
			{
				std::cout << "Error evaluating expression [" << expression << "]: " << e.GetMsg() << std::endl;
				return 0.0f;
			}
		}

		void DefineVariable(const std::string& name, double Value = 0.0, int bindingIndex = 0)
		{
			Vars[bindingIndex][name] = Value;
			Parsers[bindingIndex].DefineVar(name, &Vars[bindingIndex][name]);
		}

		void ResetTime()
		{
			deltaTime = 0.0f;
		}

	private:
		void EnsureBinding(int bindingIndex)
		{
			if (Parsers.size() == 0)
			{
				mu::Parser newParser;
				Parsers.reserve(bindingIndex + 8);
				Parsers.push_back(newParser);

				Vars.reserve(bindingIndex + 8);
				Vars.push_back({});
			}
			else if (bindingIndex >= Parsers.size())
			{
				mu::Parser newParser;
				Parsers.push_back(newParser);
				Vars.push_back({});
				std::cout << "Binding " << bindingIndex << std::endl;
			}

			if (const bool hasFmod = Parsers[bindingIndex].HasFun("fmod"); !hasFmod)
			{
				std::cout << "!hasFmod " << bindingIndex << std::endl;
				Parsers[bindingIndex].DefineFun("fmod", fmod_wrapper);
				Parsers[bindingIndex].DefineFun("pow", pow_wrapper);
				Parsers[bindingIndex].DefineFun("clamp", clamp_wrapper);
				Parsers[bindingIndex].DefineConst("_pi", 3.1415926535);
			}


		}
	};
} // NR
