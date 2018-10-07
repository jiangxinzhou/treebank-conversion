#include "CONLLReader.h"
#include "CharUtils.h"
#include "CppAssert.h"
#include "MyLib.h"

#include <sstream>
using namespace std;

namespace dparser {


	void CONLLReader::decompose_sent( Instance * const inst )
	{		
		vector<string> tokens;
		egstra::simpleTokenize(m_vecLine[0], tokens, "\t\r\n ");
		reset_sent(inst, m_vecLine.size() + 1, tokens.size());

		for (int i = 0; i < m_vecLine.size(); ++i) {
			vector<string> tokens;
			egstra::simpleTokenize(m_vecLine[i], tokens, "\t\r\n ");
			//@jiangxinzhou zhangbo
			//if (tokens.size() > 12 || tokens.size() < 8) {
			//	cerr << "\nInvalid corpus line: " << m_vecLine[i] << endl;
			//	cerr << "may need dos2unix!" << endl;
			//	inst->forms.push_back(""); // signal for bad format (Wenliang's BLLIP data has some bad cases: \t\t_);
			//	return;
			//}

			//if (tokens[3] != "_" && tokens[4] != "_" && tokens[3] != tokens[4]) {
			//	cerr << "wenliang data CPOS != POS: " << tokens[3] << " " << tokens[4] << endl;
			//}
		/*	if (tokens.size() == 12) {
				assert(inst->heads2.size() == 0 && inst->heads.size() != 0);
				inst->heads2.resize(m_vecLine.size() + 1);
				inst->heads2[0] = ROOT_HEAD;
				inst->deprels2.resize(m_vecLine.size() + 1);
				inst->deprels2[0] = NO_FORM;
			}*/

			inst->forms[i + 1] = tokens[1];
			inst->orig_lemmas[i + 1] = tokens[2];
			inst->orig_cpostags[i + 1] = tokens[3];
			inst->postags[i + 1] = tokens[3]; //modified by Daniel
			inst->orig_feats[i + 1] = tokens[5];

			if (inst->type.compare("multi-source") == 0) {
				//source
				inst->heads2[i + 1] = egstra::toInteger(tokens[6]);
				inst->deprels2[i + 1] = tokens[7];
			}
			else if (inst->type.compare("multi-target") == 0) {
				//target
				inst->heads[i + 1] = egstra::toInteger(tokens[4]);
				inst->deprels[i + 1] = tokens[5];
			}
			else {
				//target
				inst->heads[i + 1] = egstra::toInteger(tokens[4]);
				inst->deprels[i + 1] = tokens[5];
				//source
				inst->heads2[i + 1] = egstra::toInteger(tokens[6]);
				inst->deprels2[i + 1] = tokens[7];
			}

			//if (tokens.size() == 10)
			//{
			//	//if (tokens[6] == "_") inst->heads[i + 1] = -1;
			//	//else 
			//	if (_which_head == 1) {
			//		inst->heads[i + 1] = egstra::toInteger(tokens[6]);
			//		inst->deprels[i + 1] = tokens[7];
			//	}
			//	else if (_which_head == 2) {
			//		inst->heads2[i + 1] = egstra::toInteger(tokens[6]);
			//		inst->deprels2[i + 1] = tokens[7];
			//	}
			//}
			//else if (tokens.size() == 12) {
			//	//if (tokens[6] == "_") inst->heads[i + 1] = -1;
			//	//else 
			//	inst->heads[i + 1] = egstra::toInteger(tokens[6]);
			//	inst->deprels[i + 1] = tokens[7];

			//	//if (tokens[10] == "_") inst->heads[i + 1] = -1;
			//	//else 
			//	inst->heads2[i + 1] = egstra::toInteger(tokens[10]);
			//	inst->deprels2[i + 1] = tokens[11];
			//}
			//

			//if (tokens.size() > 8) {
			//	inst->pheads[i+1] = tokens[8];
			//	inst->pdeprels[i+1] = tokens[9];
			//}
		}
		inst->cpostags = inst->orig_cpostags;
	}

	void CONLLReader::reset_sent( Instance * const inst, const int length, const int token_size)
	{
		if (inst->type.compare("multi-source") == 0) {
			inst->heads2.resize(length);
			inst->heads2[0] = ROOT_HEAD;
			inst->deprels2.resize(length);
			inst->deprels2[0] = "<root>";	
		}
		else if (inst->type.compare("multi-target") == 0) {
			inst->heads.resize(length);
			inst->heads[0] = ROOT_HEAD;
			inst->deprels.resize(length);
			inst->deprels[0] = "<root>";
		}
		else { 
			inst->heads.resize(length);
			inst->heads[0] = ROOT_HEAD;
			inst->deprels.resize(length);
			inst->deprels[0] = "<root>";
			inst->heads2.resize(length);
			inst->heads2[0] = ROOT_HEAD;
			inst->deprels2.resize(length);
			inst->deprels2[0] = "<root>";
		}

		//if (token_size == 10) {
		//	if (_which_head == 1) {
		//		inst->heads.resize(length);
		//		inst->heads[0] = ROOT_HEAD;
		//		inst->deprels.resize(length);
		//		inst->deprels[0] = "<root>";
		//	}
		//	if (_which_head == 2) {
		//		inst->heads2.resize(length);
		//		inst->heads2[0] = ROOT_HEAD;
		//		inst->deprels2.resize(length);
		//		inst->deprels2[0] = "<root>";
		//	}
		//}
		//else if (token_size == 12) {
		//	inst->heads.resize(length);
		//	inst->heads[0] = ROOT_HEAD;
		//	inst->deprels.resize(length);
		//	inst->deprels[0] = "<root>";
		//	inst->heads2.resize(length);
		//	inst->heads2[0] = ROOT_HEAD;
		//	inst->deprels2.resize(length);
		//	inst->deprels2[0] = "<root>";
		//}

		inst->forms.resize(length); 
		inst->orig_lemmas.resize(length);
		inst->orig_cpostags.resize(length);
		inst->postags.resize(length);
		inst->orig_feats.resize(length);
		
		
		inst->pheads.resize(length);
		inst->pdeprels.resize(length);

		inst->forms[0] = "<root>";
		inst->orig_lemmas[0] = NO_FORM;
		inst->orig_cpostags[0] = "<root>";
		inst->postags[0] = "<root>";
		inst->orig_feats[0] = NO_FORM;
		
		
		inst->pheads[0] = NO_FORM;
		inst->pdeprels[0] = NO_FORM;
	}

	int CONLLReader::read_lines()
	{
		m_vecLine.clear();
		while (1) {
			string strLine;
			if (!my_getline(m_inf, strLine)) {
				break;
			}
			if (strLine.empty()) break;
			m_vecLine.push_back(strLine);
		}
		return m_vecLine.size();
	}

} // namespace dparser



