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
		std::map<std::string, double> Vars;
		mu::Parser parser;

		NRRules()
		{
			parser.DefineFun("fmod", fmod_wrapper);
			parser.DefineConst("_pi", 3.1415926535);
		}


		void Setup(const NRRule& rule, const NRBinding& binding, int bindingIndex, const NRModelProfile& profile, const torch::Tensor& inputTensor)
		{
			for (auto const& [name, val] : rule.Constants)
			{
				Vars[name] = val;
			}

			for (auto const& [name, val] : binding.Overrides)
			{
				Vars[name] = val;
			}

			// Inject Tensor Inputs
			for (auto const& [varName, inputList] : rule.Variables)
			{
				std::string actualInputName = (inputList.size() > bindingIndex) ? inputList[bindingIndex] : inputList[0];

				torch::Tensor val = profile.GetInputBoneValue(inputTensor, actualInputName);

				if (val.defined() && val.numel() > 0)
				{
					Vars[varName] = val.item<float>();
				}
				else
				{
					std::cerr << "[NeuraRig] ERRO: O input '" << actualInputName
							  << "' definido na regra nao foi encontrado no Schema de Inputs do JSON!" << std::endl;
					Vars[varName] = 0.0; // Valor padrão para evitar crash
				}
			}

			for (auto& [name, val] : Vars)
			{
				if (parser.GetVar().find(name) == parser.GetVar().end())
				{
					parser.DefineVar(name, &val);
				}
			}
		}

		float Eval(const std::string& expression)
		{
			try
			{
				parser.SetExpr(expression);
				float result = static_cast<float>(parser.Eval());

				// Atualizar variáveis no parser caso tenham sido modificadas (embora DefineVar deva cuidar disso)
				// Em muParser, variáveis ligadas por ponteiro são atualizadas automaticamente.
				return result;
			}
			catch (mu::Parser::exception_type& e)
			{
				std::cout << "Error evaluating expression [" << expression << "]: " << e.GetMsg() << std::endl;
				return 0.0f;
			}
		}
		
		void DefineVariable(const std::string& name, double initialValue = 0.0)
		{
			Vars[name] = initialValue;
			if (parser.GetVar().find(name) == parser.GetVar().end())
			{
				parser.DefineVar(name, &Vars[name]);
			}
		}
	};
} // NR
