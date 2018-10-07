#ifndef _PARSER_
#define _PARSER_

#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <set>
using namespace std;

#include "Driver.h"
#include "IOPipe.h"
//#include "FGen.h"
#include "common.h"
#include "Eigen/Dense"
#include "MyLib.h"
/*******************
There seems some conflicts between "ChartUtils.h" and "spthread.h".
The order of their #include can not be reversed!
I do not know why!
*******************/
#include "CharUtils.h"
#include "StringMap.h"
//#include "Options.h"
using namespace egstra;

#include "spthread.h"
#include "threadpool.h"
#include "Vocab.h"

extern sp_thread_mutex_t _global_mutex;  //@kiro
extern sp_thread_mutex_t _global_precomputation_mutex;

namespace dparser {
	/*
	this class controls the parsing process.
	*/
	class Parser
	{
	public:
		IOPipe m_pipe_train;
		IOPipe m_pipe_train2;
		IOPipe m_pipe_test;
		IOPipe m_pipe_dev;

		Options m_option;  // neural network option
		Vocab vocab;
		

		//@Daniel
		Driver m_driver;



	private:
		int _display_interval;

		string _option_file_path;  // neural network option file path
		string _dictionary_path;
		string _parameter_path;
		int _inst_max_len_to_throw;

		bool _train;
		int _iter_num;
		int _best_iter_num_so_far;
		double _best_accuracy;
		int _stop_iter_num_no_increase;

		vector<int> _inst_idx_to_read;

		bool _use_train2;
		int _inst_num_from_train2_as_one_inst;
		int _inst_num_from_train2_one_iter;
		int _inst_num_from_train1_one_iter;
		string _filename_train2;
		int _inst_max_num_train2;

		string _filename_train;
		string _filename_dev;
		string _pret_filename;

		int _inst_max_num_train;
		bool _dictionary_exist;
		bool _pamameter_exist;
		int _param_tmp_num;

		bool _test;
		string _filename_test;
		string _filename_output;
		int _param_num_for_eval;
		int _inst_max_num_eval;
		int _test_batch_size;

		/* variables used in evaluate */
		int inst_num_processed_total;
		int word_num_total;
		int word_num_dep_correct;
		int word_num_label_correct;
		int sent_num_root_corect;
		int sent_num_CM; // complete match

		int _number_processed; // total number of updates so far
		int _thread_num;

		threadpool _tp;
		static sp_thread_mutex_t _mutex;  //@kiro

		int update_iter;

		int task;

		int do_dev_per_num_batch;
	public:
		Parser() {
			//Decoder::process_options();
			//_fgen.process_options();
			process_options();
			//_tp = create_threadpool(max(1, _thread_num)); // _thread_num @kiro
			//assert(_g_thread_num > 0);
			//cerr << "Parser(): thread-num = " << _g_thread_num << endl;
		}

		~Parser(void) {
			//destroy_threadpool(_tp);
			//_tp = 0;
		}

		typedef struct thread_arg_t {
			thread_arg_t(Parser* parser, Instance * const inst, const bool is_train) : _parser(parser), _inst(inst), _is_train(is_train) { }
			Parser* const _parser;
			Instance * const _inst;
			const bool _is_train;
		};
		
		void process_options();

		void run()
		{
			if (_option_file_path != "") {  // file path read from the command line 
				m_option.load(_option_file_path);
			}
			else {
				cerr << "cannot find the neural netowrk config file!" << endl;
				exit(-1);
			}
			m_option.showOptions();

			if (_train) {
				pre_train();
				train();
				post_train();
			}
			if (_test) test(_param_num_for_eval);
		}

//		static Decoder *new_decoder(ModelParams* _modelparams, HyperParams* _hyperparams, FGen* _fgen, bool is_train, Instance *inst, Options* m_option) {
//			Decoder *decoder = new Decoder(_modelparams, _hyperparams, _fgen, is_train, inst, m_option->beam, m_option->batchSize);
//			assert(decoder);
//			return decoder;
//		}
//
//		static void delete_decoder(Decoder *&decoder) {
//			if (decoder) {
//				delete decoder;
//				decoder = 0;
//			}
//		}
//		static void delete_eval_decoder(Decoder *&decoder) {
//			if (decoder) {
//				decoder->delete_decoder();
//				delete decoder;
//				decoder = 0;
//			}
//		}
	private:

		void train();
		void train_one_iteration(const int iter_num);
		void test(const int iter);
		void get_word_pos_label_id(Instance * inst) {
			inst->forms_id.resize(inst->size());
			inst->pret_forms_id.resize(inst->size());
			inst->pos_id.resize(inst->size());
			for (int i = 0; i < inst->size(); i++) {
				int id = vocab.get_word_id(inst->forms[i]);
				inst->forms_id[i] = id;
				if (id >= vocab.words_in_train) {
					inst->forms_id[i] = 0;
				}
				inst->pret_forms_id[i] = id;
				inst->pos_id[i] = vocab.get_tag_id(inst->postags[i]);
			}

			if (inst->type.compare("multi-source") == 0) { //cdt
				inst->labels_id2.resize(inst->size());
				for (int i = 0; i < inst->size(); i++) {
					inst->labels_id2[i] = vocab.get_label_id(inst->deprels2[i]);
				}
			}
			else if (inst->type.compare("multi-target") == 0) { //hlt
				inst->labels_id.resize(inst->size());
				for (int i = 0; i < inst->size(); i++) {
					inst->labels_id[i] = vocab.get_hlt_label_id(inst->deprels[i]);
				}
			}
			else {
				inst->labels_id.resize(inst->size());
				inst->labels_id2.resize(inst->size());
				for (int i = 0; i < inst->size(); i++) {
					inst->labels_id[i] = vocab.get_hlt_label_id(inst->deprels[i]);
					inst->labels_id2[i] = vocab.get_label_id(inst->deprels2[i]);
				}
			}
		}
		void get_label(Instance * inst) {
			inst->predicted_labels.resize(inst->size());
			for (int i = 1; i < inst->size(); i++) {
				inst->predicted_labels[i] = vocab.id2hlt_label[inst->predicted_labels_id[i]];
			}
		}

	

		void set_candidate_heads_base_and_answer(Instance * inst){

			inst->set_candidate_heads_max(inst->candidate_heads_base);

			inst->set_candidate_heads_max(inst->candidate_heads_answer);
			for (int m = 1; m <= inst->size() - 1; m++) {
				int h = (inst->type.compare("multi-source") == 0) ? inst->heads2[m] : inst->heads[m];
				if (h == -1) {
					continue;
				}
				else {
					inst->set_candidate_heads_single_head_for_one_word(inst->candidate_heads_answer, m, h);
				}
			}
		}
		

		Instance *get_instance(const int inst_idx) {
			const int real_inst_idx = _inst_idx_to_read[inst_idx];
			if (real_inst_idx < m_pipe_train.getInstanceNum())
				return m_pipe_train.getInstance(real_inst_idx);
			else
				return m_pipe_train2.getInstance(real_inst_idx - m_pipe_train.getInstanceNum());
		}

		Instance *get_instance_from_train2(const int inst_idx, const int offset) {
			const int real_inst_idx = _inst_idx_to_read[inst_idx];
			assert(real_inst_idx >= m_pipe_train.getInstanceNum());
			return m_pipe_train2.getInstance(real_inst_idx - m_pipe_train.getInstanceNum() + offset);
		}

		bool is_from_train2(const int real_inst_idx) {
			return (real_inst_idx >= m_pipe_train.getInstanceNum());
		}

		void  delete_one_train_instance_after_update_gradient(Instance * const inst) {
			if (inst->id < m_pipe_train.getInstanceNum()) {
				if (m_pipe_train.use_instances_posi()) {
					delete inst;
				}
			}
			else {
				if (m_pipe_train2.use_instances_posi()) {
					delete inst;
				}
			}
		}

		// @kiro
		void delete_one_eval_instance(Instance * const inst) {
			if (m_pipe_dev.use_instances_posi()) {
				delete inst;
			}
		}
		// @kiro
		void delete_one_test_instance(Instance * const inst) {
			if (m_pipe_test.use_instances_posi()) {
				delete inst;
			}
		}

		int get_inst_num_one_iter() const { return _inst_idx_to_read.size(); }
		void prepare_train_instances();
		void pre_train();
		void post_train() {
			m_pipe_train.dealloc_instance();
			m_pipe_train.closeInputFile();
			if (_use_train2) {
				m_pipe_train2.dealloc_instance();
				m_pipe_train2.closeInputFile();
			}
			m_pipe_dev.dealloc_instance();
		}


		void evaluate(IOPipe &pipe, const bool is_test);
		void reset_evaluate_metrics();
		void output_evaluate_metrics();

		void load_dictionaries() {
			//_fgen.load_dictionaries(_dictionary_path);
			vocab.load(".");
			cerr << "load dict done!" << endl;
		}


		void save_parameters() 
		{
			switch (task) 
			{
			case 0:
				// m_driver._modelparams.multi_saveModel();
				if (_use_train2)
				{
					/*
					多任务 保存 word_emb, tag_emb, lstm模型
					*/
					m_driver._modelparams.saveModel_embed_bilstm();
				}
				break;
			case 1:
				m_driver._modelparams.sp_ex_saveModel();
				break;
			}
		}
			
		void save_pret_parameters() {
			m_driver._modelparams.saveModel_embed_bilstm();
		}

		/*void load_parameters(const int iter) {
			switch (task) {
			case 0:
				break;
			case 1:
				m_driver._modelparams.sp_loadModel();
				break;
			case 2:
				m_driver._modelparams.pattern_loadModel();
				break;
			case 3:
				break;
			case 4:
				break;
			}
		}*/

		void delete_parameters(const int iter) {
		}
		void evaluate_one_instance(const Instance * const inst);
		void error_num_dp(const Instance *inst, const bool bIncludePunc, int &nDepError, int &nLabelError, int &nUnscoredToken, bool &bRootCorrect) const;
	};
}


#endif

