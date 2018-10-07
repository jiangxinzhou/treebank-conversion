#ifndef _DRIVER_H_
#define _DRIVER_H_

#include <iostream>

#include "HyperParams.h"
#include "Instance.h"
#include "threadpool.h"
#include "ComputionGraph.h"


class Driver {
public:
	Driver() {}

	~Driver() {}
public:
	Graph _cg;
	vector<GraphBuilder> _builders;
	ModelParams _modelparams;
	HyperParams _hyperparams;
	ModelUpdate _ada;  // model update

	int task;

	threadpool tp;
	int thread_num;
public:
	void process_options() 
	{
		task = -1;
		int tmp;
		if (options::get("task", tmp)) {
			task = tmp;
		}
		thread_num = 5;  // thread num
		if (options::get("thread-num", tmp)) { assert(tmp > 0); thread_num = tmp; }
	}

	inline void initial(int max_sent_len)
	{
		if (!_hyperparams.bValid()) {
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}

		_builders.resize(_hyperparams.batch);
		for (int i = 0; i < _hyperparams.batch; i++) {
			_builders[i].modelparams = &(_modelparams);
			_builders[i].hyperparams = &(_hyperparams);
		
		}
		GraphBuilder::max_sentence_length = max_sent_len; 
		if (thread_num > 1) {
			tp = create_threadpool(thread_num);
		}

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}

	typedef struct thread_arg_t {
		thread_arg_t(int word_num, vector<GraphBuilder*>* builder, vector<dparser::Instance*> *insts) :
			_word_num(word_num), _builder(builder), _insts(insts) {}

		int _word_num;
		vector<GraphBuilder*> *_builder;
		vector<dparser::Instance*>* _insts;
	};


	static void run_one_thread(void *arg) {
		vector<GraphBuilder*> *_builder = ((thread_arg_t*)arg)->_builder;
		int _word_num = ((thread_arg_t*)arg)->_word_num;
		vector<dparser::Instance*> *_insts = ((thread_arg_t*)arg)->_insts;

		Graph *cg = new Graph();
		cg->clearValue(true);
		int count = _insts->size();
		//cout << "count: " << count << endl;
		for (int i = 0; i < count; i++) {
			(*_builder)[i]->forward(cg, (*_insts)[i]);
		}
		cg->compute();

		for (int i = 0; i < count; i++) {
			CRFLoss* loss = new CRFLoss();
			if ((*_insts)[i]->type.compare("multi-source") == 0 || (*_insts)[i]->type.compare("multi-target") == 0
				|| (*_insts)[i]->type.compare("synchronization") == 0) {

				loss->loss( &( (*_builder)[i]->arc_biaffine ), (*_insts)[i], _word_num);
				loss->rel_loss(& ( (*_builder)[i]->rel_biaffine ), (*_insts)[i], _word_num);
			}
			else {
				loss->loss((*_builder)[i]->arc_biaffines, (*_insts)[i], _word_num);
				loss->rel_loss((*_builder)[i]->rel_biaffines, (*_insts)[i], _word_num);
			}
			
			loss->dealloc();
			delete loss;
		}

		cg->backward();

		//delete loss;
		delete cg;
		delete ((thread_arg_t*)arg);
	}

	typedef struct pred_thread_arg_t {
		pred_thread_arg_t(vector<GraphBuilder*>* builder, vector<dparser::Instance*> *insts) :
			_builder(builder), _insts(insts) {}

		vector<GraphBuilder*> *_builder;
		vector<dparser::Instance*>* _insts;
	};


	static void pred_run_one_thread(void *arg) {

		vector<GraphBuilder*> *_builder = ((pred_thread_arg_t*)arg)->_builder;
		vector<dparser::Instance*> *_insts = ((pred_thread_arg_t*)arg)->_insts;
		



		Graph *cg = new Graph();
		cg->clearValue(false);
		int count = _insts->size();
		//cout << "count: " << count << endl;
		for (int i = 0; i < count; i++) {
			//cout << "type: "; cout << (*_insts)[i]->type << endl;
			//cout << "i: " << i << endl;
			(*_builder)[i]->forward(cg, (*_insts)[i]);
		}

		cg->compute();


		for (int i = 0; i < count; i++) {
			CRFLoss* loss = new CRFLoss();
			(*_insts)[i]->predicted_heads.resize((*_insts)[i]->size());
			(*_insts)[i]->predicted_labels_id.resize((*_insts)[i]->size());
			if ((*_insts)[i]->type.compare("multi-source") == 0 || (*_insts)[i]->type.compare("multi-target") == 0
				|| (*_insts)[i]->type.compare("synchronization") == 0) {


				loss->predict_arc_and_rel( &( (*_builder)[i]->arc_biaffine ), &( (*_builder)[i]->rel_biaffine ), (*_insts)[i]);
	
			}
			else {
			
				loss->predict_arc_and_rel((*_builder)[i]->arc_biaffines, (*_builder)[i]->rel_biaffines, (*_insts)[i]);
			}
			delete loss;
		}
		
		delete cg;
		delete ((pred_thread_arg_t*)arg);
	}

	typedef struct cons_pred_thread_arg_t {
		cons_pred_thread_arg_t(vector<GraphBuilder*>* builder, vector<dparser::Instance*> *insts) :
			_builder(builder), _insts(insts) {}

		vector<GraphBuilder*> *_builder;
		vector<dparser::Instance*>* _insts;
	};

	static void cons_pred_run_one_thread(void *arg) {

		vector<GraphBuilder*> *_builder = ((cons_pred_thread_arg_t*)arg)->_builder;
		vector<dparser::Instance*> *_insts = ((cons_pred_thread_arg_t*)arg)->_insts;

		Graph *cg = new Graph();
		cg->clearValue(false);
		int count = _insts->size();
		//cout << "count: " << count << endl;
		for (int i = 0; i < count; i++) {
			//cout << "type: "; cout << (*_insts)[i]->type << endl;
			//cout << "i: " << i << endl;

			(*_builder)[i]->forward(cg, (*_insts)[i]);
		}

		cg->compute();

		for (int i = 0; i < count; i++) {
			CRFLoss* loss = new CRFLoss();
			(*_insts)[i]->predicted_heads.resize((*_insts)[i]->size());
			(*_insts)[i]->predicted_labels_id.resize((*_insts)[i]->size());
			loss->constrained_predict((*_builder)[i]->arc_biaffines, (*_builder)[i]->rel_biaffines, (*_insts)[i]);
			delete loss;
		}
		delete cg;
		delete ((cons_pred_thread_arg_t*)arg);
	}

	inline dtype train(const vector<dparser::Instance*> &insts) {
		dtype cost = 0.0;

		if (thread_num > 1) 
		{
			int insts_num = insts.size();
			if (insts_num > _builders.size()) 
			{
				std::cout << "input example number larger than predefined batch number" << std::endl;
				return 1000;
			}

			int word_num = 0;
			for (int count = 0; count < insts_num; count++) 
			{
				word_num += insts[count]->size() - 1;
			}

			vector<vector<dparser::Instance*>> vec_insts;
			vec_insts.clear();
			vec_insts.resize(thread_num);
			vector<vector<GraphBuilder*>> vec_builders;
			vec_builders.clear();
			vec_builders.resize(thread_num);

			
			for (int idx = 0, p = 0; idx < insts_num;p++) 
			{
				for (int idy = 0; idy < (200 / thread_num) && idx < insts_num; idy++, idx++) 
				{
					vec_insts[p].push_back(insts[idx]);
					vec_builders[p].push_back(&_builders[idx]);
				}
				thread_arg_t *arg = new thread_arg_t(word_num, &vec_builders[p], &vec_insts[p]);
				dispatch_threadpool(tp, run_one_thread, (void*)(arg));
			}

			wait_all_jobs_done(tp);
		}
		else {
			int insts_num = insts.size();

			int word_num = 0;

			_cg.clearValue(true);
			for (int count = 0; count < insts_num; count++) 
			{
				//forward
				word_num += insts[count]->size() - 1;
				_builders[count].forward(&_cg, insts[count]);
			}

			_cg.compute();
			

			assert(task >= 0);

			switch (task) 
			{
			case 0:
				for (int count = 0; count < insts_num; count++) 
				{
					cost += _modelparams.loss->loss(&_builders[count].arc_biaffine, insts[count], word_num);
					cost += _modelparams.loss->rel_loss(&_builders[count].rel_biaffine, insts[count], word_num);
				}
				break;
			case 1:
				for (int count = 0; count < insts_num; count++)
				{
					cost += _modelparams.loss->loss(_builders[count].arc_biaffines, insts[count], word_num);
					cost += _modelparams.loss->rel_loss(_builders[count].rel_biaffines, insts[count], word_num);
				}
				break;
			}
			
			_cg.backward();
		}

		return cost;
	}

	inline void predict(vector<dparser::Instance*> &insts) {
		if (thread_num > 1) {
			int insts_num = insts.size();
			vector<vector<dparser::Instance*>> vec_insts;
			vec_insts.clear();
			vec_insts.resize(thread_num);
			vector<vector<GraphBuilder*>> vec_builders;
			vec_builders.clear();
			vec_builders.resize(thread_num);

			for (int idx = 0, p = 0; idx < insts_num; p++) {
				for (int idy = 0; idy < (200 / thread_num) && idx < insts_num; idy++, idx++) {
					vec_insts[p].push_back(insts[idx]);
					vec_builders[p].push_back(&_builders[idx]);
				}
				pred_thread_arg_t *arg = new pred_thread_arg_t(&vec_builders[p], &vec_insts[p]);
				dispatch_threadpool(tp, pred_run_one_thread, (void*)(arg));
			}
			wait_all_jobs_done(tp);
		}
		else {
			int insts_num = insts.size();

			_cg.clearValue();
			for (int idx = 0; idx < insts_num; idx++) {
				_builders[idx].forward(&_cg, insts[idx]);
			}

			_cg.compute();
		
			assert(task >= 0);

			switch (task) {
			case 0:
				for (int idx = 0; idx < insts_num; idx++) {
					insts[idx]->predicted_heads.resize(insts[idx]->size());
					insts[idx]->predicted_labels_id.resize(insts[idx]->size());
					//_modelparams.loss.predict_arc_and_rel(&_builders[idx].arc_biaffine, &_builders[idx].rel_biaffine, insts[idx]);
					_modelparams.loss->predict_arc_and_rel(&_builders[idx].arc_biaffine, &_builders[idx].rel_biaffine, insts[idx]);
				}
				break;
			case 1:
				for (int idx = 0; idx < insts_num; idx++) {
					insts[idx]->predicted_heads.resize(insts[idx]->size());
					insts[idx]->predicted_labels_id.resize(insts[idx]->size());
					_modelparams.loss->predict_arc_and_rel(_builders[idx].arc_biaffines, _builders[idx].rel_biaffines, insts[idx]);
				}
				break;
			}
		
		}
	}

	inline void constrained_predict(vector<dparser::Instance*> &insts) {
		assert(thread_num > 0);
		assert(task >= 0);
		int insts_num = insts.size();
		assert(insts_num <= 200);
		vector<vector<dparser::Instance*>> vec_insts;
		vec_insts.clear();
		vec_insts.resize(thread_num);
		vector<vector<GraphBuilder*>> vec_builders;
		vec_builders.clear();
		vec_builders.resize(thread_num);

		for (int idx = 0, p = 0; idx < insts_num; p++) {
			for (int idy = 0; idy < (200 / thread_num) && idx < insts_num; idy++, idx++) {
				vec_insts[p].push_back(insts[idx]);
				vec_builders[p].push_back(&_builders[idx]);
			}
			cons_pred_thread_arg_t *arg = new cons_pred_thread_arg_t(&vec_builders[p], &vec_insts[p]);
			dispatch_threadpool(tp, cons_pred_run_one_thread, (void*)(arg));
		}
		wait_all_jobs_done(tp);
	}

	void updateModel() {
		_ada.updateAdam();
		_ada.gradClip(_hyperparams.clip);
		//_ada.rescaleGrad(1.0 / word_num);
		_ada.m_update();
	}

private:
	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

	


#endif
