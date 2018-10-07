#ifndef _DEP_PIPE_
#define _DEP_PIPE_

#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using namespace std;

#include "Instance.h"
#include "CONLLReader.h"
#include "CONLLWriter.h"
#include "common.h"
#include "N3LDG.h"

#include "Util-options.h"
using namespace egstra;

namespace dparser {

	class IOPipe
	{
	private:
		vector<int> m_vecInstIdxToRead;
		vector<Instance *> m_instances;
		vector<size_t> m_instances_posi;
		size_t _start_id;

		string m_inf_name;
		string m_outf_name;
		CONLLReader *m_reader;
		CONLLWriter *m_writer;

		bool _filtered_arc;
		bool _use_instances_posi;
		size_t _inf_current_posi;
		bool _use_constrained_search_space;

		bool _copy_cpostag_from_postag;
		bool _get_cpostag_from_pdeprel;
		bool _english;
		bool _use_lemma;
		bool _labeled;

		CRFLoss check;

	public:
		string _type;
	public:
		IOPipe() : m_reader(0), m_writer(0) {}

		~IOPipe()
		{
			dealloc_instance();
			closeInputFile();
			closeOutputFile();
		}

		//@jiangxinzhou zhangbo
		/*void set_which_head(int h) {
			m_reader->_which_head = h;
		}*/
		void set_type(const string& type) {
			_type = type;
		}

		void dealloc_instance() {
			for (int i = 0; i < m_instances.size(); ++i) {
				assert(m_instances[i]);
				delete m_instances[i];
				m_instances[i] = 0;
			}
			m_instances_posi.clear();
			m_instances.clear();
			m_vecInstIdxToRead.clear();
		}

		const string &in_file_name() const {
			return m_inf_name;
		}

		void use_instances_posi(bool flag) { _use_instances_posi = flag; }
		bool use_instances_posi() const { return _use_instances_posi; }

		void set_filtered_arc_flag(bool flag) { _filtered_arc = flag; }

		void process_options() {
			_use_instances_posi = true;
			_inf_current_posi = 0;
			_filtered_arc = false;
			_use_constrained_search_space = false;
			_labeled = true;

			int tmp;
			if(options::get("use-file-position-when-read-instance", tmp)) {
				_use_instances_posi = (1 == tmp);
			}

			if(options::get("labeled", tmp)) {
				_labeled = (1 == tmp);
			}

			_copy_cpostag_from_postag = false;
			_get_cpostag_from_pdeprel = false;
			if(options::get("copy-cpostag-from-postag", tmp)) {
				_copy_cpostag_from_postag = (1 == tmp);
			}
			if(options::get("get-cpostag-from-pdeprel", tmp)) {
				_get_cpostag_from_pdeprel = (1 == tmp);
			}

			_english = true;
			_use_lemma = false;
			if(options::get("english", tmp)) {
				_english = tmp;
			}
			if(options::get("use-lemma", tmp)) {
				_use_lemma = tmp;
			}

			if(options::get("use-constrained-search-space", tmp)) {
				_use_constrained_search_space = (1 == tmp);
			}
		}

		const string &input_filename() { return m_inf_name; }
		const string &output_filename() { return m_outf_name; }

		int openInputFile(const char *filename) {
			m_inf_name = filename;
			m_reader = new CONLLReader();
			m_reader->_type = _type;
			if (!m_reader) {
				string str = "IOPipe::IOPipe() create reader error";
				cerr << str << endl;
				throw(str);
			}
			_inf_current_posi = 0;
			return m_reader->openFile(filename); 
		}

		void closeInputFile() {	
			if (m_reader) {
				m_reader->closeFile();
				delete m_reader;
				m_reader = 0;
			}
		}

		int openOutputFile(const char *filename) { 
			m_outf_name = filename;
			m_writer = new CONLLWriter();
			if (!m_writer) {
				string str = "IOPipe::IOPipe() create writer error";
				cerr << str << endl;
				throw(str);
			}
			return m_writer->openFile(filename);
		}

		void closeOutputFile() { 
			if (m_writer) {
				m_writer->closeFile(); 
				delete m_writer;
				m_writer = 0;
			}
		}

		void getInstancesFromInputFile(const int startId = 0, const int maxInstNum=-1, const int instMaxLen=-1, const int instMinLen=0);

		void shuffleTrainInstances() {
			random_shuffle(m_vecInstIdxToRead.begin(), m_vecInstIdxToRead.end());
		}

		void preprocessInstance( Instance *inst );

		int getInstanceNum() const {
			return _use_instances_posi ? m_instances_posi.size() : m_instances.size();
		}

		Instance *getInstance(const int instIdx) {
			if (instIdx < 0 || instIdx >= m_vecInstIdxToRead.size()) {
				cerr << "\nIOPipe::getInstance instIdx range err: " << instIdx << endl;
				return 0;
			}
			const int id = m_vecInstIdxToRead[instIdx];
			const int global_id = _start_id + id;
			if (_use_instances_posi) {
				size_t posi = m_instances_posi[ id ];
				Instance *inst = m_reader->getNext(global_id, posi);
				preprocessInstance(inst);
				return inst;
			} else {
				return m_instances[ id ];
			}
		}

		void fillVecInstIdxToRead() {
			m_vecInstIdxToRead.clear();
			m_vecInstIdxToRead.resize(getInstanceNum());
			for (int i = 0; i < getInstanceNum(); ++i) m_vecInstIdxToRead[i] = i;
		}

		int writeInstance(const Instance *inst) {
			return m_writer->write(inst);
		}
	};
}

#endif


