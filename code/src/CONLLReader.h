#ifndef _CONLL_READER_
#define _CONLL_READER_

#pragma once

#include <fstream>
#include <iostream>
using namespace std;

#include "Instance.h"
#include "common.h"


namespace dparser {

	/*
	this class reads conll-format data (10 columns, no srl-info)
	*/

	class CONLLReader
	{
	public:
		CONLLReader() { _type = "multi-target"; }
		~CONLLReader() {}

/*		Instance *getNext(const int id) {
			const int word_num = read_lines();
			if (word_num > 0) {
				Instance *inst = new Instance(id);
				decompose_sent(inst);
				return inst;
			} else {
				return 0;
			}
		}
*/
		Instance *getNext(const int id, size_t &posi) {
			if (m_inf.rdstate() != ios::goodbit) {
				m_inf.clear();
			}
			m_inf.seekg(posi, ios::beg);
			const int word_num = read_lines();
			posi = m_inf.tellg();
			if (word_num > 0) {
				Instance *inst = new Instance(id);
				//jxz
				inst->type = _type;

				decompose_sent(inst);
				return inst;
			} else {
				return 0;
			}
		}
	protected:
		void reset_sent(Instance * const inst, const int length, const int token_size);

		void decompose_sent(Instance * const inst);

		// return the number of words in the sentence, excluding W0
		int read_lines();

	public:

		int openFile(const char *filename) {
			if (m_inf.is_open()) {
				m_inf.close();
			}
			m_inf.open(filename, std::ios::binary);

			if (!m_inf.is_open()) {
				cerr << "CoNLLReader::openFile() err: " << filename << endl;
				return -1;
			}

			return 0;
		}

		void closeFile() {
			if (m_inf.is_open()) {
				m_inf.close();
			}
		}

	public:
		//int _which_head;
		string _type;
	protected:
		ifstream m_inf;
		vector<string> m_vecLine;
	};

} // namespace dparser


#endif

