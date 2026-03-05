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
		std::vector<std::unordered_map<std::string, double>> Vars;
		std::vector<mu::Parser> Parsers;

		NRRules() = default;

		void Setup(const NRRule& rule, int bindingIndex)
		{
			EnsureBinding(bindingIndex);

			for (auto const& [name, val] : rule.Constants)
				DefineVariable(name, val, bindingIndex);

			// Vars que serão preenchidas a cada sample (vindas do Tensor)
			for (auto const& [varName, _inputList] : rule.Variables)
				DefineVariable(varName, 0.0, bindingIndex);

			// Vars derivadas (Logic)
			for (auto const& [logicName, _expr] : rule.Logic)
				DefineVariable(logicName, 0.0, bindingIndex);

			// Vars de saída das phases (offset_x/y/z, progress, etc)
			for (auto const& phase : rule.Phases)
				for (auto const& [formulaName, _expr] : phase.Formulas)
					DefineVariable(formulaName, 0.0, bindingIndex);

			DefineVariable("t_one", 0.0, bindingIndex);
			DefineVariable("t_two", 0.0, bindingIndex);
			DefineVariable("offset_y", 0.0, bindingIndex);
		}

		void SetTensorInputs(int bindingIndex, const NRRule& rule, const NRModelProfile& profile, const torch::Tensor& currentInput)
		{
			EnsureBinding(bindingIndex);

			for (auto const& [varName, inputList] : rule.Variables)
			{
				if(inputList.empty())
				{
					continue;
				}

				auto pick = inputList[0];
				if(inputList.size() > bindingIndex)
				{
					 pick = inputList[bindingIndex];
				}
				std::cout << "DEBUG: " << varName << " = " << pick << std::endl;
				std::cout << "DEBUG: " << varName << " = " << profile.GetInputBoneValue(currentInput, pick).item<double>() << std::endl;
				Vars[bindingIndex][varName] = profile.GetInputBoneValue(currentInput, pick).item<double>();
			}
		}

		float Eval(int bindingIndex, const std::string& expression)
		{
			try
			{
				EnsureBinding(bindingIndex);
				Parsers[bindingIndex].SetExpr(expression);
				return static_cast<float>(Parsers[bindingIndex].Eval());
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

		static double GetInputScalar(const NRModelProfile& profile, const torch::Tensor& currentInput, const std::string& token)
		{
			std::string name = token;
			int componentIndex = -1;

			const auto dotPos = token.find('.');
			if (dotPos != std::string::npos)
			{
				name = token.substr(0, dotPos);
				componentIndex = std::atoi(token.substr(dotPos + 1).c_str());
			}

			for (const auto& block : profile.Inputs)
			{
				if (block.Name != name)
					continue;

				const int idx = (componentIndex >= 0) ? componentIndex : 0;
				if (idx < 0 || idx >= block.FloatCount)
					return 0.0;

				const int64_t absolute = static_cast<int64_t>(block.Offset + idx);
				return currentInput[absolute].item<double>();
			}

			return 0.0;
		}
	};
} // NR
