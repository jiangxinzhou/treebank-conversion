#include "Parser.h"
#include <cstdio>
#include <iomanip>
#include <ctime>
#include <cfloat>
#include <algorithm>
using namespace std;

sp_thread_mutex_t _global_mutex;
sp_thread_mutex_t _global_precomputation_mutex;



namespace dparser {

	sp_thread_mutex_t Parser::_mutex;
	//double Parser::lossSum;
	void Parser::process_options()
	{
		m_pipe_train.process_options();
		m_pipe_train2.process_options();
		m_pipe_test.process_options();
		m_pipe_dev.process_options();
		//@jiangxinzhou zhangbo
		m_driver._modelparams.process_options();
		m_driver.process_options();

		_train = false;
		_test = false;
		_inst_max_len_to_throw = 150;
		_inst_max_num_eval = -1;
		_inst_max_num_train = -1;
		_inst_max_num_train2 = -1;
		_test_batch_size = 10000;
		do_dev_per_num_batch = -1;

		_display_interval = 100;

		_dictionary_path = ".";
		_parameter_path = ".";

		_use_train2 = false;
		_filename_train2 = "train2.conll06";
		_inst_num_from_train2_one_iter = -1;
		_inst_num_from_train1_one_iter = -1;
		_inst_max_num_train2 = -1;
		_inst_num_from_train2_as_one_inst = 1;
		_filename_train = "";
		_filename_dev = "";
		_iter_num = 20;
		_stop_iter_num_no_increase = 30;

		_dictionary_exist = false;
		_pamameter_exist = false;
		_param_tmp_num = -1;

		_filename_test = "";
		_filename_output = "";
		_param_num_for_eval = -1;

		task = -1;

		int tmp; string strtmp;	double dtmp;
		_thread_num = 5;  // thread num
		if (options::get("thread-num", tmp)) { assert(tmp > 0); _thread_num = tmp; }

		if (options::get("train", tmp)) {
			_train = tmp;
		}
		if (options::get("test", tmp)) {
			_test = tmp;
		}

		if (options::get("task", tmp)) {
			task = tmp;
		}

		

		if (_train) {
			
			

			if (options::get("stop-iter-num-no-increase", tmp)) {
				_stop_iter_num_no_increase = tmp;
			}
			if (options::get("do-dev-per-num-batch", tmp)) {
				do_dev_per_num_batch = tmp;
			}

			if (options::get("inst-num-from-train-1-one-iter", tmp)) {
				_inst_num_from_train1_one_iter = tmp;
			}

			if (options::get("use-train-2", tmp)) {
				_use_train2 = (1 == tmp);
			}
			if (_use_train2) {
				if (options::get("train-file-2", strtmp)) {
					_filename_train2 = strtmp;
				}
				if (options::get("inst-max-num-train-2", tmp)) {
					_inst_max_num_train2 = tmp;
				}
				if (options::get("inst-num-from-train-2-one-iter", tmp)) {
					_inst_num_from_train2_one_iter = tmp;
				}
			}
		}

		/*if (options::get("lstm-layer-num", tmp)) {
			assert(tmp == 2 || tmp == 3);
			lstm_layer_num = tmp;
		}*/
		
		if (options::get("test-batch-size", tmp)) {
			_test_batch_size = tmp;
		}
		

		if (options::get("inst-max-len-to-throw", tmp)) {
			_inst_max_len_to_throw = tmp;
		}

		if (options::get("inst-max-num-train", tmp)) {
			_inst_max_num_train = tmp;
		}
		if (options::get("inst-max-num-eval", tmp)) {
			_inst_max_num_eval = tmp;
		}

		if (options::get("display-interval", tmp)) {
			_display_interval = tmp;
		}

		if (options::get("dictionary-path", strtmp)) {
			_dictionary_path = strtmp;
		}
		if (options::get("parameter-path", strtmp)) {
			_parameter_path = strtmp;
		}
		if (options::get("option_file_path", strtmp)) {
			_option_file_path = strtmp;
		}

		if (options::get("dictionary-exist", tmp)) {
			_dictionary_exist = tmp;
		}

		if (options::get("parameter-exist", tmp)) {
			_pamameter_exist = tmp;
		}
		if (options::get("param-tmp-num", tmp)) {
			_param_tmp_num = tmp;
			if (_param_tmp_num <= 0) _param_tmp_num = 1;
		}

		if (options::get("train-file", strtmp)) {
			_filename_train = strtmp;
		}
		if (options::get("dev-file", strtmp)) {
			_filename_dev = strtmp;
		}
		if (options::get("pret-file", strtmp)) {
			_pret_filename = strtmp;
		}
		if (options::get("iter-num", tmp)) {
			_iter_num = tmp;
		}

		if (options::get("test-file", strtmp)) {
			_filename_test = strtmp;
		}
		if (options::get("output-file", strtmp)) {
			_filename_output = strtmp;
		}
		if (options::get("param-num-for-eval", tmp)) {
			_param_num_for_eval = tmp;
		}
		if (options::get("filtered-arc-train1", tmp)) {
			m_pipe_train.set_filtered_arc_flag((1 == tmp));
		}
		if (options::get("filtered-arc-train2", tmp)) {
			m_pipe_train2.set_filtered_arc_flag((1 == tmp));
		}
		if (options::get("filtered-arc-test", tmp)) {
			m_pipe_test.set_filtered_arc_flag((1 == tmp));
		}
	}

	void Parser::prepare_train_instances()
	{
		_inst_idx_to_read.clear();

		m_pipe_train.shuffleTrainInstances();
		const int inst_num_train1 = m_pipe_train.getInstanceNum();
		const int real_inst_num_train1_used_one_iter =
			(_inst_num_from_train1_one_iter > 0 && _inst_num_from_train1_one_iter < inst_num_train1) ? _inst_num_from_train1_one_iter : inst_num_train1;
		for (int i = 0; i < real_inst_num_train1_used_one_iter; ++i) {	// use the first-n1/n2 instances  of each corpus
			_inst_idx_to_read.push_back(i);
		}
		if (_use_train2) {
			m_pipe_train2.shuffleTrainInstances();
			const int inst_num_train2 = m_pipe_train2.getInstanceNum();
			const int real_inst_num_train2_used_one_iter =
				(_inst_num_from_train2_one_iter > 0 && _inst_num_from_train2_one_iter < inst_num_train2) ? _inst_num_from_train2_one_iter : inst_num_train2;
			for (int i = 0; i < real_inst_num_train2_used_one_iter; ++i) {
				_inst_idx_to_read.push_back(inst_num_train1 + i);	// if idx >= inst_num_train1, then it comes from corpus 2.
			}
		}
		cerr << "instance num from train1: " << real_inst_num_train1_used_one_iter << endl;
		if (_use_train2) {
			cerr << "instance num from train2: " << _inst_idx_to_read.size() - real_inst_num_train1_used_one_iter << endl;
			cerr << "instance num total: " << _inst_idx_to_read.size() << endl;
		}
		random_shuffle(_inst_idx_to_read.begin(), _inst_idx_to_read.end());	// randomize the instances from two corpus
	}

	void Parser::train()
	{
		cerr << "train begining..." << endl;
		_best_iter_num_so_far = 0;
		_best_accuracy = 0.;

		_number_processed = 0;
		update_iter = 0;
		int iter = 1;

		int batch_count = 0;
		int best_batch_count = 0;

		for (; iter <= _iter_num; iter++) 
		{
			cerr << "\n***** Iteration #" << iter << " *****"; print_time();
			prepare_train_instances();
			cerr << "preprocess instance done" << endl;

			
			const int inst_num = get_inst_num_one_iter();
			const bool is_train = true;
			for (int i = 0; i < inst_num;) 
			{
					
				int j = 0;
				vector<Instance *> inst_list;
				inst_list.reserve(0);
				for (; j < m_option.batchSize; j++, i++)
				{ //m_option.batchSize @kiro
					if (i >= inst_num) break;
					Instance *inst = get_instance(i);
					get_word_pos_label_id(inst);
					set_candidate_heads_base_and_answer(inst);
					inst_list.push_back(inst);
				}
				m_driver.train(inst_list);

				for (int idx = 0; idx < inst_list.size(); idx++) 
				{
					delete_one_train_instance_after_update_gradient(inst_list[idx]);
				}

				m_driver.updateModel();
		

				update_iter++;
				m_driver._ada._alpha = m_driver._hyperparams.adaAlpha * pow(0.75, update_iter*1.0 / 5000);

				batch_count++;

				if (i % _display_interval == 0) cerr << i << " ";
				if (i % (_display_interval * 10) == 0) print_time();

				if (batch_count % do_dev_per_num_batch == 0 || i == inst_num)
				{
					cerr << "\n*** decoding: on " << m_pipe_dev.in_file_name() << " [it=" << iter << "]"; print_time();
					reset_evaluate_metrics();
					evaluate(m_pipe_dev, false);
					cerr << "\n done "; print_time();
					output_evaluate_metrics();

					double this_accuracy = 100.0 * word_num_label_correct / word_num_total;
					//double this_accuracy = 100.0 * word_num_dep_correct / word_num_total;

					vector<int> del;
					if (this_accuracy > _best_accuracy + 1e-5)
					{
						if (_best_iter_num_so_far > 0) 
						{
							del.push_back(_best_iter_num_so_far);
						}
						best_batch_count = batch_count;
						_best_iter_num_so_far = iter;
						_best_accuracy = this_accuracy;

						// test on testset
						cerr << "\n*** decoding: on " << m_pipe_test.in_file_name() << " [it=" << iter << "]"; print_time();
						reset_evaluate_metrics();
						evaluate(m_pipe_test, false);
						cerr << "\n done "; print_time();
						output_evaluate_metrics();

						save_parameters();
					}

					batch_count = 0;

					if (_best_iter_num_so_far > 0) 
					{
						cerr << "\nbest LAS so far (a): " << _best_accuracy << " [it = " << _best_iter_num_so_far << "]" << endl;
					}

				
					if (_best_iter_num_so_far + _stop_iter_num_no_increase < iter) 
					{
						cerr << "\n\n ---- STOP training due to no accuracy increase in many iterations!" << endl;
						exit(0);
					}

					
				}
			}
		}
	}


	void Parser::train_one_iteration(const int iter_num)
	{
		const int inst_num = get_inst_num_one_iter();
		const bool is_train = true;
		for (int i = 0; i < inst_num;) {
			
			int j = 0;
			vector<Instance *> inst_list;
			inst_list.reserve(0);
			for (; j < m_option.batchSize; j++, i++) { //m_option.batchSize @kiro
				if (i >= inst_num) break;
				Instance *inst = get_instance(i);
				get_word_pos_label_id(inst);
				set_candidate_heads_base_and_answer(inst);
				inst_list.push_back(inst);
			}
			m_driver.train(inst_list);

			
			for (int idx = 0; idx < inst_list.size(); idx++) {
				delete_one_train_instance_after_update_gradient(inst_list[idx]);
			}
			
			m_driver.updateModel();
			//m_driver._modelparams.m_saveModel2();
			
			update_iter++;
			m_driver._ada._alpha = m_driver._hyperparams.adaAlpha * pow(0.75, update_iter*1.0 / 5000); 

			

			if (i % _display_interval == 0) cerr << i << " ";
			if (i % (_display_interval * 10) == 0) print_time();
		}

	}

	void Parser::pre_train()
	{
		if (!_dictionary_exist) 
		{
			int tmp, min_occur_count;
			if (options::get("min-occur-count", tmp)) 
			{
				min_occur_count = tmp;
			}

			//vocab.create_pattern_dict(_filename_train, 0);

			if (!_use_train2) 
			{
				vocab.init(_filename_train, _pret_filename, min_occur_count);
			}
			else
			{
				vector<string> train_filenames;
				train_filenames.push_back(_filename_train);
				train_filenames.push_back(_filename_train2);
				vocab.init(train_filenames, _pret_filename, min_occur_count);
			}

			cout << "------------ create dictionary done --------------" << endl;

			vocab.save(".");

			cout << "------------ save dictionary done -----------------" << endl;
			exit(0);
		}
		load_dictionaries();

/*--------------------------------task--------------------------------------------------------*/
		assert(task >= 0);
		vector<vector<double>> tmp_embs;
		vocab.get_pret_embs(tmp_embs, _pret_filename);
		m_driver._hyperparams.setRequared(m_option);

		if (_train) 
		{
			switch (task) 
			{
			case 0:
				m_pipe_train.set_type("multi-target");
				m_pipe_train2.set_type("multi-source");
				m_pipe_dev.set_type("multi-target");
				m_pipe_test.set_type("multi-target");

				m_driver._modelparams.multi_initial(vocab, tmp_embs, m_driver._hyperparams);
				m_driver._modelparams.exportModelParams(m_driver._ada);
				break;
			case 1:
				m_pipe_train.set_type("conversion");
				m_pipe_dev.set_type("conversion");
				m_pipe_test.set_type("conversion");

				m_driver._modelparams.sp_ex_pattern_fix_initial(vocab, tmp_embs, m_driver._hyperparams);
				m_driver._modelparams.exportModelParams_sp_ex_pattern_fix_conv(m_driver._ada);
				m_driver._modelparams.loadModel_embed_bilstm();
				break;
			}
		}


		if (_test) 
		{
			switch (task) 
			{
			case 0:
				m_pipe_test.set_type("multi-target");
				m_driver._modelparams.multi_initial(vocab, tmp_embs, m_driver._hyperparams);
				m_driver._modelparams.multi_loadModel();
				break;
			case 1:
				m_pipe_test.set_type("conversion");
				m_driver._modelparams.sp_ex_pattern_fix_initial(vocab, tmp_embs, m_driver._hyperparams);
				m_driver._modelparams.sp_ex_loadModel();
				break;
			}
		}

		m_driver.initial(_inst_max_len_to_throw);

/*--------------------------------IOpipe read file-----------------------------------------------*/

		m_pipe_train.openInputFile(_filename_train.c_str());
		std::cout << "train1-jz: " << _filename_train << std::endl;

		//m_pipe_train.set_type("multi-target");
		m_pipe_train.getInstancesFromInputFile(0, _inst_max_num_train, _inst_max_len_to_throw);

		if (_use_train2) {
			const int inst_num_train1 = m_pipe_train.getInstanceNum();
			m_pipe_train2.openInputFile(_filename_train2.c_str());
			//m_pipe_train2.set_type("multi-source");
			std::cout << "train2-jz: " << _filename_train2 << std::endl;
			m_pipe_train2.getInstancesFromInputFile(inst_num_train1, _inst_max_num_train2, _inst_max_len_to_throw, 0);
		}

		m_pipe_dev.use_instances_posi(false);
		m_pipe_dev.openInputFile(_filename_dev.c_str());
		std::cout << "dev-jz: " << _filename_dev << std::endl;
		//m_pipe_dev.set_type("multi-target");
		m_pipe_dev.getInstancesFromInputFile(0, _inst_max_num_eval, _inst_max_len_to_throw);
		m_pipe_dev.closeInputFile();

		m_pipe_test.use_instances_posi(false);
		m_pipe_test.openInputFile(_filename_test.c_str());
		std::cout << "test-jz: " << _filename_test << std::endl;
		//m_pipe_dev.set_type("multi-target");
		m_pipe_test.getInstancesFromInputFile(0, _inst_max_num_eval, _inst_max_len_to_throw);
		m_pipe_test.closeInputFile();
	}

	void Parser::evaluate(IOPipe &pipe, const bool is_test)
	{
		/*if (pipe.use_instances_posi()) {
			cerr << "do not use_instances_posi for test/dev data: " << pipe.input_filename() << endl;
			exit(-1);
		}*/

		const int inst_num = pipe.getInstanceNum();
		inst_num_processed_total += inst_num;
		const bool is_train = false;

		/*for (int i = 0; i < inst_num; ++i) {
		Instance *inst = pipe.getInstance(i);
		get_word_pos_label_id(inst);
		}*/

	
		for (int i = 0; i < inst_num; ) {
			vector<Instance*> insts;
			insts.reserve(0);
			for (int j = 0; j < m_option.batchSize; ++i, ++j) {
				if (i >= inst_num) break;
				Instance *inst = pipe.getInstance(i);
				get_word_pos_label_id(inst);
				set_candidate_heads_base_and_answer(inst);
				insts.push_back(inst);
			}

			if (_test) {
				m_driver.predict(insts);
				//m_driver.constrained_predict(insts);
			}
			else {
				m_driver.predict(insts);
			}


			for (int idx = 0; idx < insts.size(); idx++) {
				get_label(insts[idx]);
				if (_test) {
					m_pipe_test.writeInstance(insts[idx]);
					evaluate_one_instance(insts[idx]);
				}
				else {
					evaluate_one_instance(insts[idx]);
				}
				if (pipe.use_instances_posi()) {
					delete insts[idx];
				}
			}

			if (i % _display_interval == 0) cerr << i << " ";
			if (i % (_display_interval * 10) == 0) print_time();
		}

		

		/*for (int i = 0; i < inst_num; ++i) {
			Instance *inst = pipe.getInstance(i);
			get_label(inst);
			if(_test) m_pipe_test.writeInstance(inst);
			else {
				evaluate_one_instance(inst);
			}
		}*/

	
		cerr << "\ninstance num: " << inst_num; print_time();
	}

	void Parser::reset_evaluate_metrics() {
		inst_num_processed_total = 0;
		word_num_total = 0;
		word_num_label_correct = 0;
		word_num_dep_correct = 0;
		sent_num_root_corect = 0;
		sent_num_CM = 0;
	}

	void Parser::output_evaluate_metrics() {
		cerr.precision(5);
		if (inst_num_processed_total > 0) cerr << "CM:    \t" << sent_num_CM << "/" << inst_num_processed_total << " = " << sent_num_CM*100.0 / inst_num_processed_total << endl;
		if (inst_num_processed_total > 0) cerr << "ROOT:  \t" << sent_num_root_corect << "/" << inst_num_processed_total << " = " << sent_num_root_corect*100.0 / inst_num_processed_total << endl;
		if (word_num_total > 0)           cerr << "UAS:   \t" << word_num_dep_correct << "/" << word_num_total << " = " << word_num_dep_correct*100.0 / word_num_total << endl;
		if (word_num_total > 0)           cerr << "LAS:   \t" << word_num_label_correct << "/" << word_num_total << " = " << word_num_label_correct*100.0 / word_num_total << endl;
	}

	void Parser::test(const int iter)
	{
		assert(iter >= 1);
		cerr << "\n\n eval: " << iter; print_time();

		//m_pipe_test.use_instances_posi(false);
		m_pipe_test.openInputFile(_filename_test.c_str());
		m_pipe_test.openOutputFile(_filename_output.c_str());

		if (!_train) {
			load_dictionaries();
		}

		m_pipe_test.getInstancesFromInputFile(0, _inst_max_num_eval, _inst_max_len_to_throw);

		//evaluate(m_pipe_test, true);
		//load_parameters(iter);
		cout <<"load param done" << endl;
		reset_evaluate_metrics();

		evaluate(m_pipe_test, true);

		cerr << "done";  print_time();
		output_evaluate_metrics();

		m_pipe_test.closeInputFile();
		m_pipe_test.closeOutputFile();

	/*	int start_id = 0;
		while (1) {
			const int inst_num_left = _inst_max_num_eval < 0 ? _test_batch_size : (_inst_max_num_eval - inst_num_processed_total);
			if (inst_num_left <= 0) break;

			m_pipe_test.getInstancesFromInputFile(start_id,
				_test_batch_size < inst_num_left ? _test_batch_size : inst_num_left,
				_inst_max_len_to_throw);

			if (m_pipe_test.getInstanceNum() <= 0) break;

			start_id += m_pipe_test.getInstanceNum();

			evaluate(m_pipe_test, true);
		}*/
		
	}

	bool pair_int_double_compare(const pair<int, double> &a, const pair<int, double> &b) {
		return a.second < b.second;
	}

	void output(const vector< pair<int, double> > &vec_pair) {
		for (int i = 0; i < vec_pair.size(); ++i) {
			cerr << vec_pair[i].first << " " << vec_pair[i].second << endl;
		}
	}

	int get_int_at_least_one(double d) {
		const int i = (int)(d);
		const double x = d - (double)(i);
		if (x > 0.5) {
			return (i + 1);
		}
		else {
			return max(1, i);
		}
	}

	void Parser::error_num_dp(const Instance *inst, const bool bIncludePunc, int &nDepError, int &nLabelError, int &nUnscoredToken, bool &bRootCorrect) const
	{
		nDepError = 0;
		nLabelError = 0;
		bRootCorrect = false;
		nUnscoredToken = 0;
		if (inst->type.compare("multi-source") == 0) {
			for (int i = 1; i < inst->size(); ++i) {
				if (inst->heads2[i] == -1) {
					++nUnscoredToken;
					continue;
				}

				if (0 == inst->predicted_heads[i] && 0 == inst->heads2[i]) bRootCorrect = true;
				if (inst->predicted_heads[i] != inst->heads2[i]) {
					++nDepError;
					++nLabelError;
				}
				else if (inst->predicted_labels_id[i] != inst->labels_id2[i]) {
					++nLabelError;
				}
			}
		}
		else {
			for (int i = 1; i < inst->size(); ++i) {
				if (inst->heads[i] == -1) {
					++nUnscoredToken;
					continue;
				}

				if (0 == inst->predicted_heads[i] && 0 == inst->heads[i]) bRootCorrect = true;
				if (inst->predicted_heads[i] != inst->heads[i]) {
					++nDepError;
					++nLabelError;
				}
				else if (inst->predicted_labels_id[i] != inst->labels_id[i]) {
					++nLabelError;
				}
			}
		}
	}

	void Parser::evaluate_one_instance(const Instance * const inst)
	{
		const int length = inst->size();

		bool bRootCorrect;
		int nDepError, nLabelError, nUnscoredToken;
		error_num_dp(inst, false, nDepError, nLabelError, nUnscoredToken, bRootCorrect);
		word_num_total += length - 1 - nUnscoredToken;
		word_num_dep_correct += length - 1 - nUnscoredToken - nDepError;
		word_num_label_correct += length - 1 - nUnscoredToken - nLabelError;
		if (bRootCorrect) ++sent_num_root_corect;
		if (0 == nDepError && 0 == nLabelError) ++sent_num_CM;
	}

} // namespace dparser



