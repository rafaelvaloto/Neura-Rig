// Project: NeuralRig
// Copyright (c) 2026 Rafael Valoto
// All rights reserved.
#pragma once
#include "muParser.h"
#include "NRTypes.h"

static mu::value_type fmod_wrapper(mu::value_type v1, mu::value_type v2) {
	return std::fmod(v1, v2);
}

namespace NR
{
	class NRRules
	{
	public:
		double deltaTime = 0.0;
		std::vector<std::unordered_map<std::string, double>> Vars;
		std::vector<mu::Parser> Parsers;

		NRRules() = default;

		void Setup(const NRRule& rule, int bindingIndex)
		{
			EnsureBinding(bindingIndex);

			for (auto const& [name, val] : rule.Constants)
			{
				DefineVariable(name, val, bindingIndex);
			}


			// Vars que serão preenchidas a cada sample (vindas do Tensor)
			for (auto const& [varName, _inputList] : rule.Variables)
			{
				DefineVariable(varName, 0.0, bindingIndex);
			}


			// Vars derivadas (Logic)
			for (auto const& [logicName, _expr] : rule.Logic)
			{
				DefineVariable(logicName, 0.0, bindingIndex);
			}

			// Vars de saída das phases (offset_x/y/z, progress, etc)
			for (auto const& phase : rule.Phases)
			{
				for (auto const& [formulaName, _expr] : phase.Formulas)
				{
					DefineVariable(formulaName, 0.0, bindingIndex);
				}
			}
		}

		void SetTensorInputs(int bindingIndex, const NRRule& rule, const NRModelProfile& profile, const torch::Tensor& currentInput)
		{
			EnsureBinding(bindingIndex);

			for (auto const& [varName, inputList] : rule.Variables)
			{
				auto pick = inputList[0];
				if(inputList.size() > bindingIndex)
				{
					 pick = inputList[bindingIndex];
				}

				if (pick == "delta_time")
				{
					if (bindingIndex == 0)
					{
						deltaTime += profile.GetInputBoneValue(currentInput, pick).item<double>();
					}

					Vars[bindingIndex][varName] = deltaTime;
					continue;
				}
				Vars[bindingIndex][varName] = profile.GetInputBoneValue(currentInput, pick).item<double>();
			}
		}

		double Eval(int bindingIndex, const std::string& expression)
		{
			try
			{
				EnsureBinding(bindingIndex);
				Parsers[bindingIndex].SetExpr(expression);
				return Parsers[bindingIndex].Eval();
			}
			catch (mu::Parser::exception_type& e)
			{
				std::cout << "Error evaluating expression [" << expression << "]: " << e.GetMsg() << std::endl;
				return 0.0f;
			}
		}

		void DefineVariable(const std::string& name, double initialValue = 0.0, int bindingIndex = 0)
		{
			EnsureBinding(bindingIndex);

			if (!Vars[bindingIndex].contains(name))
			{
				Vars[bindingIndex][name] = initialValue;
				Parsers[bindingIndex].DefineVar(name, &Vars[bindingIndex][name]); // <-- ligação REAL com o muParser
			}
		}

		void ResetTime()
		{
			if (deltaTime >= 4.0)
			{
				deltaTime = 0.0f;
			}
		}

	private:
		void EnsureBinding(int bindingIndex)
		{
			if (bindingIndex < 0) return;

			if (static_cast<size_t>(bindingIndex) >= Vars.size())
				Vars.resize(bindingIndex + 1);

			if (static_cast<size_t>(bindingIndex) >= Parsers.size())
			{
				Parsers.resize(bindingIndex + 1);
			}

			if (const bool hasFmod = Parsers[bindingIndex].HasFun("fmod"); !hasFmod)
			{
				Parsers[bindingIndex].DefineFun("fmod", fmod_wrapper);
				Parsers[bindingIndex].DefineConst("_pi", 3.1415926535);
			}
		}

	};
} // NR
