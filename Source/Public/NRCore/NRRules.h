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
		std::map<std::string, float> Vars;
		mu::Parser parser;

		NRRules()
		{
			parser.DefineFun("fmod", fmod_wrapper);
			parser.DefineConst("PI", 3.14159265f);
		}


		void Setup(const NRRule& rule, const NRBinding& binding, int bindingIndex, const NRModelProfile& profile, const torch::Tensor& inputTensor)
		{
			for (auto const& [name, val] : rule.Constants)
			{
				Vars[name] = static_cast<float>(val);
			}

			for (auto const& [name, val] : binding.Overrides)
			{
				Vars[name] = static_cast<float>(val);
			}

			// Inject Tensor Inputs
			for (auto const& [varName, inputList] : rule.Variables)
			{
				std::string actualInputName = (inputList.size() > bindingIndex) ? inputList[bindingIndex] : inputList[0];

				torch::Tensor val = profile.GetInputBoneValue(inputTensor, actualInputName);
				Vars[varName] = val.item<float>();
			}

			for (auto& [name, val] : Vars)
			{
				parser.DefineVar(name, reinterpret_cast<mu::value_type*>(&val));
			}
		}

		float Eval(const std::string& expression)
		{
			try
			{
				parser.SetExpr(expression);
				return static_cast<float>(parser.Eval());
			}
			catch (mu::Parser::exception_type& e)
			{
				std::cout << "Error evaluating expression: " << e.GetMsg() << std::endl;
				return 0.0f;
			}
		}
	};
} // NR
