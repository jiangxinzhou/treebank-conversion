#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "FeatureDictionary.h"
#include "HyperParams.h"
#include "StringMap.h"
#include "ComputionGraph.h"
#include "./util/Util-options.h"
#include "N3LDG.h"
#include "Vocab.h"
#include "Parser.h"

using namespace egstra;
using namespace dparser;

// Each model consists of two parts, building neural graph and defining output losses.

class ModelParams{			
public:
	//@jiangxinzhou zhangbo
	bool _use_train2;

	Alphabet wordAlpha; 
	Alphabet posAlpha;

	LookupTable pret_word_embs;  //表b 大
	LookupTable word_embs; //表a 小
	LookupTable tag_embs; 

	LookupTable pattern_emb_param;
	LookupTable dist_emb_param;
	//new pattern
	LookupTable expand_pattern_param;
	LookupTable m_label_emb;
	LookupTable h_label_emb;
	LookupTable lcn_label_emb;

	LSTM1Params lstm_param;
	LSTM1Params lstm_param_right;
	LSTM1Params lstm_param1;
	LSTM1Params lstm_param_right1;
	LSTM1Params lstm_param2;
	LSTM1Params lstm_param_right2;

	UniParams mlp_head_param;
	UniParams mlp_dep_param;
	
	UniParams mlp_head_rel_param;
	UniParams mlp_dep_rel_param;
	
	BiaffineParams biaffine_arc;
	BiaffineParams biaffine_rel;

//  multi-source params
	UniParams mlp_head2_param;
	UniParams mlp_dep2_param;

	UniParams mlp_head_rel2_param;
	UniParams mlp_dep_rel2_param;

	BiaffineParams biaffine_arc2;
	BiaffineParams biaffine_rel2;

	// for conversion
	UniParams conv_mlp_head_param;
	UniParams conv_mlp_dep_param;
	TreeLSTMParams tree_lstm_param;
	TreeLSTMParams tree_lstm_param_top_down;
	SBiaffineParam sbiaffine_arc;
	vector<SBiaffineParam> sbiaffine_rel;

	//rel conversion
	UniParams  conv_mlp_head_rel_param;
	UniParams  conv_mlp_dep_rel_param;

	LookupTable label_emb;

	//
	UniParams src_mlp_param;
	UniParams src_mlp_rel_param;

	ParseLogitsLoss logits_loss;
	CRFLoss crf_loss;
	BaseLoss* loss;
	string loss_type;
	
	int tree_lstm_size;
public:
	//@jiangxinzhou zhangbo
	void process_options() {
		int tmp;
		if (options::get("use-train-2", tmp)) {
			_use_train2 = (1 == tmp);
		}

		string loss_type;
		assert(options::get("loss-type", loss_type));
		if (loss_type.compare("logits-loss") == 0) {
			loss = &logits_loss;
		}
		else if (loss_type.compare("crf-loss") == 0) {
			loss = &crf_loss;
		}
	}


	bool multi_initial(const Vocab& vocab, const vector<vector<double>> &tmp_embs, HyperParams& h_params) {
		//assert(m_option.wordFile != "");
		pret_word_embs.initial_constant(vocab.word_size(), h_params.word_emb_size, false);
		pret_word_embs.getE(tmp_embs);

		word_embs.initial_constant(vocab.words_in_train, h_params.word_emb_size, true);
		tag_embs.initial(vocab.tag_size(), h_params.tag_emb_size, true);

		int target_labelSize = vocab.hlt_label_size() - 2;
		int source_labelSize = vocab.label_size() - 2;

		lstm_param.initial(h_params.lstm_output_size, h_params.word_emb_size + h_params.tag_emb_size);
		lstm_param_right.initial(h_params.lstm_output_size, h_params.word_emb_size + h_params.tag_emb_size);
		lstm_param1.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param_right1.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param2.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param_right2.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);

		mlp_head_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2, true);
		mlp_dep_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2, true);

		mlp_head_rel_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2, true);
		mlp_dep_rel_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2, true);

		biaffine_arc.initial(h_params.mlp_size + 1, h_params.mlp_size, false, 1);
		biaffine_rel.initial(h_params.mlp_rel_size + 2, h_params.mlp_rel_size + 2, false, target_labelSize);

		//multi-source
		mlp_head2_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2, true);
		mlp_dep2_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2, true);

		mlp_head_rel2_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2, true);
		mlp_dep_rel2_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2, true);

		biaffine_arc2.initial(h_params.mlp_size + 1, h_params.mlp_size, false, 1);
		biaffine_rel2.initial(h_params.mlp_rel_size + 2, h_params.mlp_rel_size + 2, false, source_labelSize);

		return true;
	}

	void multi_saveModel() {
		ofstream out("models.bin");

		word_embs.save(out);
		tag_embs.save(out);

		lstm_param.save(out);
		lstm_param_right.save(out);
		lstm_param1.save(out);
		lstm_param_right1.save(out);
		if (lstm_layer_num == 3) {
			lstm_param2.save(out);
			lstm_param_right2.save(out);
		}

		mlp_dep_param.save(out);
		mlp_head_param.save(out);
		mlp_dep_rel_param.save(out);
		mlp_head_rel_param.save(out);

		biaffine_arc.save(out);
		biaffine_rel.save(out);

		// no source param load
	}

	void multi_loadModel() {
		ifstream in("models.bin");

		word_embs.load(in);
		tag_embs.load(in);

		lstm_param.load(in);
		lstm_param_right.load(in);
		lstm_param1.load(in);
		lstm_param_right1.load(in);
		if (lstm_layer_num == 3) {
			lstm_param2.load(in);
			lstm_param_right2.load(in);
		}

		mlp_dep_param.load(in);
		mlp_head_param.load(in);
		mlp_dep_rel_param.load(in);
		mlp_head_rel_param.load(in);

		biaffine_arc.load(in);
		biaffine_rel.load(in);

		// no source param load
	}

	void exportModelParams(ModelUpdate& ada) {
		word_embs.exportAdaParams(ada);
		tag_embs.exportAdaParams(ada);

		lstm_param.exportAdaParams(ada);
		lstm_param_right.exportAdaParams(ada);
		lstm_param1.exportAdaParams(ada);
		lstm_param_right1.exportAdaParams(ada);
		lstm_param2.exportAdaParams(ada);
		lstm_param_right2.exportAdaParams(ada);

		mlp_head_param.exportAdaParams(ada);
		mlp_dep_param.exportAdaParams(ada);
		mlp_head_rel_param.exportAdaParams(ada);
		mlp_dep_rel_param.exportAdaParams(ada);

		biaffine_arc.exportAdaParams(ada);
		biaffine_rel.exportAdaParams(ada);


		mlp_head2_param.exportAdaParams(ada);
		mlp_dep2_param.exportAdaParams(ada);
		mlp_head_rel2_param.exportAdaParams(ada);
		mlp_dep_rel2_param.exportAdaParams(ada);

		biaffine_arc2.exportAdaParams(ada);
		biaffine_rel2.exportAdaParams(ada);
	}

	bool sp_ex_pattern_fix_initial(const Vocab& vocab, const vector<vector<double>> &tmp_embs, HyperParams& h_params) {
		tree_lstm_size = h_params.tree_lstm_output_size;

		pret_word_embs.initial_constant(vocab.word_size(), h_params.word_emb_size, false);
		pret_word_embs.getE(tmp_embs);

		word_embs.initial_constant(vocab.words_in_train, h_params.word_emb_size, true);
		tag_embs.initial(vocab.tag_size(), h_params.tag_emb_size, true);

		int target_labelSize = vocab.hlt_label_size() - 2;
		int source_labelSize = vocab.label_size() - 2;


		label_emb.initial(vocab.label_size(), h_params.label_emb_size, true);

		lstm_param.initial(h_params.lstm_output_size, h_params.word_emb_size + h_params.tag_emb_size);
		lstm_param_right.initial(h_params.lstm_output_size, h_params.word_emb_size + h_params.tag_emb_size);
		lstm_param1.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param_right1.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param2.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);
		lstm_param_right2.initial(h_params.lstm_output_size, h_params.lstm_output_size * 2);

		tree_lstm_param.initial(h_params.tree_lstm_output_size, h_params.lstm_output_size * 2 + h_params.label_emb_size, true);
		tree_lstm_param_top_down.initial(h_params.tree_lstm_output_size, h_params.lstm_output_size * 2 + h_params.label_emb_size, false);

		conv_mlp_head_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2 + h_params.tree_lstm_output_size * 3 + h_params.pattern_emb_size + h_params.m_label_emb_size + h_params.h_label_emb_size + h_params.lcn_label_emb_size, true);
		conv_mlp_dep_param.initial(h_params.mlp_size, h_params.lstm_output_size * 2 + h_params.tree_lstm_output_size * 3 + h_params.pattern_emb_size + h_params.m_label_emb_size + h_params.h_label_emb_size + h_params.lcn_label_emb_size, true);

		conv_mlp_head_rel_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2 + h_params.tree_lstm_output_size * 3 + h_params.pattern_emb_size + h_params.m_label_emb_size + h_params.h_label_emb_size + h_params.lcn_label_emb_size, true);
		conv_mlp_dep_rel_param.initial(h_params.mlp_rel_size, h_params.lstm_output_size * 2 + h_params.tree_lstm_output_size * 3 + h_params.pattern_emb_size + h_params.m_label_emb_size + h_params.h_label_emb_size + h_params.lcn_label_emb_size, true);

		sbiaffine_arc.initial(h_params.mlp_size + 1 ,
			h_params.mlp_size, false);

		sbiaffine_rel.resize(target_labelSize);
		for (int i = 0; i < target_labelSize; i++) {
			sbiaffine_rel[i].initial(h_params.mlp_rel_size + 2, h_params.mlp_rel_size + 2, false);
		}

		expand_pattern_param.initial(h_params.range.size() + 5, h_params.pattern_emb_size, true);

		m_label_emb.initial(vocab.label_size(), h_params.m_label_emb_size, true);
		h_label_emb.initial(vocab.label_size(), h_params.h_label_emb_size, true);
		lcn_label_emb.initial(vocab.label_size(), h_params.lcn_label_emb_size, true);
	}
	
	void exportModelParams_sp_ex_pattern_fix_conv(ModelUpdate& ada) {
		expand_pattern_param.exportAdaParams(ada);

	
		m_label_emb.exportAdaParams(ada);
		h_label_emb.exportAdaParams(ada);
		lcn_label_emb.exportAdaParams(ada);
		


		label_emb.exportAdaParams(ada);
		conv_mlp_dep_rel_param.exportAdaParams(ada);
		conv_mlp_head_rel_param.exportAdaParams(ada);
		conv_mlp_head_param.exportAdaParams(ada);
		conv_mlp_dep_param.exportAdaParams(ada);

		tree_lstm_param.exportAdaParams(ada);
		tree_lstm_param_top_down.exportAdaParams(ada);

		sbiaffine_arc.exportAdaParams(ada);

		for (int i = 0; i < sbiaffine_rel.size(); i++) {
			sbiaffine_rel[i].exportAdaParams(ada);
		}
	}

	void sp_ex_saveModel() {
		ofstream out("models.bin");

		//pret_word_embs.save(out);
		word_embs.save(out);
		tag_embs.save(out);


		lstm_param.save(out);
		lstm_param_right.save(out);
		lstm_param1.save(out);
		lstm_param_right1.save(out);

		if (expand_pattern_param.nDim > 0) {

			expand_pattern_param.save(out);
		}

		if (lcn_label_emb.nDim > 0) {
			lcn_label_emb.save(out);
		}

		if (m_label_emb.nDim > 0) {
			m_label_emb.save(out);
		}

		if (h_label_emb.nDim > 0) {
			h_label_emb.save(out);
		}


		if (tree_lstm_size > 0) {
			tree_lstm_param.save(out);
			tree_lstm_param_top_down.save(out);
			if (label_emb.nDim > 0) {
				label_emb.save(out);
			}
		}


		conv_mlp_head_rel_param.save(out);
		conv_mlp_dep_rel_param.save(out);
		conv_mlp_head_param.save(out);
		conv_mlp_dep_param.save(out);

		sbiaffine_arc.save(out);
		for (int i = 0; i < sbiaffine_rel.size(); i++) {
			sbiaffine_rel[i].save(out);
		}
	}

	void sp_ex_loadModel() {
		ifstream in("models.bin");

		//pret_word_embs.save(out);
		word_embs.load(in);
		tag_embs.load(in);


		lstm_param.load(in);
		lstm_param_right.load(in);
		lstm_param1.load(in);
		lstm_param_right1.load(in);

		if (expand_pattern_param.nDim > 0) {
			expand_pattern_param.load(in);
		}

		if (lcn_label_emb.nDim > 0) {
			lcn_label_emb.load(in);
		}

		if (m_label_emb.nDim > 0) {
			m_label_emb.load(in);

		}

		if (h_label_emb.nDim > 0) {
			h_label_emb.load(in);
		}


		if (tree_lstm_size > 0) {
			tree_lstm_param.load(in);
			tree_lstm_param_top_down.load(in);
			if (label_emb.nDim > 0) {
				label_emb.load(in);
			}
		}


		conv_mlp_head_rel_param.load(in);
		conv_mlp_dep_rel_param.load(in);
		conv_mlp_head_param.load(in);
		conv_mlp_dep_param.load(in);

		sbiaffine_arc.load(in);
		for (int i = 0; i < sbiaffine_rel.size(); i++) {
			sbiaffine_rel[i].load(in);
		}
	}

	void loadModel_embed_bilstm() {
		ifstream in("embed-bilstm");

		word_embs.load(in);
		tag_embs.load(in);

		lstm_param.load(in);
		lstm_param_right.load(in);
		lstm_param1.load(in);
		lstm_param_right1.load(in);
		if (lstm_layer_num == 3) {
			lstm_param2.load(in);
			lstm_param_right2.load(in);
		}
	}

	void saveModel_embed_bilstm() {
		ofstream out("embed-bilstm");

		word_embs.save(out);
		tag_embs.save(out);

		lstm_param.save(out);
		lstm_param_right.save(out);
		lstm_param1.save(out);
		lstm_param_right1.save(out);
		if (lstm_layer_num == 3) {
			lstm_param2.save(out);
			lstm_param_right2.save(out);
		}


		//mlp_head_rel_param.save(out);

		//biaffine_rel.save(out);
	}
};

#endif
