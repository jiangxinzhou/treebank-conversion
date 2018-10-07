#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "Instance.h"

// Each model consists of two parts, building neural graph and defining output losses.
class  GraphBuilder {
public:
	static int max_sentence_length;

public:
	ModelParams* modelparams;
	HyperParams* hyperparams;
	int length;

	vector<LookupNode> word_pret_emb;
	vector<LookupNode> word_rand_emb;
	vector<PAddNode> word_emb;
	vector<DropNode> word_emb_drop;

	vector<LookupNode> tag_emb;
	vector<DropNode> tag_emb_drop;
	vector<ConcatNode> emb_input;

	LSTM1Builder lstm_build;
	LSTM1Builder lstm_build_right;
	vector<ConcatNode> lstm_output;

	LSTM1Builder lstm_build1;
	LSTM1Builder lstm_build_right1;
	vector<ConcatNode> lstm_output1;

	vector<DropNode> lstm_drop;

	LSTM1Builder lstm_build2;
	LSTM1Builder lstm_build_right2;
	vector<ConcatNode> lstm_output2;

	vector<UniNode> mlp_head;
	vector<DropNode> mlp_head_drop;
	vector<UniNode> mlp_dep;
	vector<DropNode> mlp_dep_drop;
	vector<UniNode> mlp_head_rel;
	vector<DropNode> mlp_head_rel_drop;
	vector<UniNode> mlp_dep_rel;
	vector<DropNode> mlp_dep_rel_drop;
	BiaffineNode arc_biaffine;
	BiaffineNode rel_biaffine;

	//for task 10
	vector<UniNode> mlp_head2;
	vector<DropNode> mlp_head_drop2;
	vector<UniNode> mlp_dep2;
	vector<DropNode> mlp_dep_drop2;
	vector<UniNode> mlp_head_rel2;
	vector<DropNode> mlp_head_rel_drop2;
	vector<UniNode> mlp_dep_rel2;
	vector<DropNode> mlp_dep_rel_drop2;
	BiaffineNode arc_biaffine2;
	BiaffineNode rel_biaffine2;
	//@ jiangxinzhou zhangbo
	//for label sp

   //for tree conversion
	vector<vector<TreeLSTMBuilder>> tree_lstm_build;
	vector<vector<TreeLSTMBuilder>> tree_lstm_build_top_down;
	vector<vector<ConcatNode>> concat_sp;
	vector<vector<ConcatNode>> concat_head;
	vector<vector<ConcatNode>> concat_dep;
	vector<vector<UniNode>> conv_mlp_head;
	vector<vector<DropNode>> conv_mlp_head_drop;
	vector<vector<UniNode>> conv_mlp_dep;
	vector<vector<DropNode>> conv_mlp_dep_drop;
	//for label sp-tree
	vector<LookupNode> label_emb;
	vector<ConcatNode> sp_input;
	vector<vector<vector<SBiaffineNode>>> rel_biaffines;  // [dep][head][label]
	vector<vector<UniNode>> conv_mlp_head_rel;;
	vector<vector<DropNode>> conv_mlp_head_rel_drop;
	vector<vector<UniNode>> conv_mlp_dep_rel;
	vector<vector<DropNode>> conv_mlp_dep_rel_drop;

	//for pattern distance conversion
	vector<vector<LookupNode>> pattern_emb;
	vector<vector<ConcatNode>> biaffine_inputs_dep;
	vector<vector<ConcatNode>> biaffine_inputs_head;

	vector<vector<SBiaffineNode>> arc_biaffines;


	//for dist
	vector<vector<LookupNode>> dist_emb;

	//for expand pattern
	vector<vector<LookupNode>> expand_pattern_emb;
	vector<LookupNode> m_label_emb;
	vector<LookupNode> h_label_emb;
	vector<LookupNode> lcn_label_emb;
	vector<vector<ConcatNode>> biaffine_inputs_dep_rel;
	vector<vector<ConcatNode>> biaffine_inputs_head_rel;

public:
	GraphBuilder() {}

	~GraphBuilder() {clear();}

	void clear() {
		length = -1;
		mlp_head2.clear();
		mlp_head_drop2.clear();
		mlp_dep2.clear();
		mlp_dep_drop2.clear();
		mlp_head_rel2.clear();
		mlp_head_rel_drop2.clear();
		mlp_dep_rel2.clear();
		mlp_dep_rel_drop2.clear();

		lcn_label_emb.clear();
		expand_pattern_emb.clear();
		m_label_emb.clear();
		h_label_emb.clear();
		biaffine_inputs_dep_rel.clear();
		biaffine_inputs_head_rel.clear();

		word_pret_emb.clear();
		word_rand_emb.clear();
		word_emb.clear();
		word_emb_drop.clear();
		tag_emb.clear();
		tag_emb_drop.clear();
		emb_input.clear();

		lstm_build.clear();
		lstm_build_right.clear();
		lstm_output.clear();

		lstm_build1.clear();
		lstm_build_right1.clear();
		lstm_output1.clear();

		lstm_drop.clear();

		lstm_build2.clear();
		lstm_build_right2.clear();
		lstm_output2.clear();

		mlp_head.clear();
		mlp_head_drop.clear();
		mlp_dep.clear();
		mlp_dep_drop.clear();
		mlp_head_rel.clear();
		mlp_head_rel_drop.clear();
		mlp_dep_rel.clear();
		mlp_dep_rel_drop.clear();


		tree_lstm_build.clear();
		tree_lstm_build_top_down.clear();
		concat_sp.clear();
		concat_head.clear();
		concat_dep.clear();
		conv_mlp_head.clear();
		conv_mlp_head_drop.clear();
		conv_mlp_dep.clear();
		conv_mlp_dep_drop.clear();

		conv_mlp_head_rel.clear();
		conv_mlp_head_rel_drop.clear();
		conv_mlp_dep_rel.clear();
		conv_mlp_dep_rel_drop.clear();


		pattern_emb.clear();
		dist_emb.clear();
		biaffine_inputs_dep.clear();
		biaffine_inputs_head.clear();

		arc_biaffines.clear();
		rel_biaffines.clear();

		label_emb.clear();
		sp_input.clear();
	}

	// multi 
	void resize(int len) {
		length = len;

		word_pret_emb.resize(len);
		word_rand_emb.resize(len);
		word_emb.resize(len);
		word_emb_drop.resize(len);

		tag_emb.resize(len);
		tag_emb_drop.resize(len);
		emb_input.resize(len);

		lstm_build.resize(len);
		lstm_build_right.resize(len);
		lstm_output.resize(len);


		lstm_build1.resize(len);
		lstm_build_right1.resize(len);
		lstm_output1.resize(len);

		lstm_drop.resize(len);
		lstm_build2.resize(len);
		lstm_build_right2.resize(len);
		lstm_output2.resize(len);

		//target
		mlp_head.resize(len);
		mlp_head_drop.resize(len);
		mlp_dep.resize(len);
		mlp_dep_drop.resize(len);
		mlp_head_rel.resize(len);
		mlp_head_rel_drop.resize(len);
		mlp_dep_rel.resize(len);
		mlp_dep_rel_drop.resize(len);
	}

	void multi_target_setParams() {
		for (int idx = 0; idx < length; idx++) {
			word_pret_emb[idx].setParam(&(modelparams->pret_word_embs));
			word_rand_emb[idx].setParam(&(modelparams->word_embs));
			tag_emb[idx].setParam(&(modelparams->tag_embs));

			mlp_head[idx].setParam(&(modelparams->mlp_head_param));
			mlp_dep[idx].setParam(&(modelparams->mlp_dep_param));
			mlp_head_rel[idx].setParam(&(modelparams->mlp_head_rel_param));
			mlp_dep_rel[idx].setParam(&(modelparams->mlp_dep_rel_param));


		}

		arc_biaffine.setParam(&(modelparams->biaffine_arc), 1, 0);
		rel_biaffine.setParam(&(modelparams->biaffine_rel), 2, 2);

		lstm_build.init(&(modelparams->lstm_param), 0.33, 0.33, true);//0.33
		lstm_build_right.init(&(modelparams->lstm_param_right), 0.33, 0.33, false);//0.33
		lstm_build1.init(&(modelparams->lstm_param1), 0.33, 0.33, true);//0.33
		lstm_build_right1.init(&(modelparams->lstm_param_right1), 0.33, 0.33, false);//0.33
		lstm_build2.init(&(modelparams->lstm_param2), 0.33, 0.33, true);//0.33
		lstm_build_right2.init(&(modelparams->lstm_param_right2), 0.33, 0.33, false);//0.33
	}

	void multi_source_setParams() {
		for (int idx = 0; idx < length; idx++) {
			word_pret_emb[idx].setParam(&(modelparams->pret_word_embs));
			word_rand_emb[idx].setParam(&(modelparams->word_embs));
			tag_emb[idx].setParam(&(modelparams->tag_embs));

			mlp_head[idx].setParam(&(modelparams->mlp_head2_param));
			mlp_dep[idx].setParam(&(modelparams->mlp_dep2_param));
			mlp_head_rel[idx].setParam(&(modelparams->mlp_head_rel2_param));
			mlp_dep_rel[idx].setParam(&(modelparams->mlp_dep_rel2_param));
		}


		arc_biaffine.setParam(&(modelparams->biaffine_arc2), 1, 0);
		rel_biaffine.setParam(&(modelparams->biaffine_rel2), 2, 2);

		lstm_build.init(&(modelparams->lstm_param), 0.33, 0.33, true);//0.33
		lstm_build_right.init(&(modelparams->lstm_param_right), 0.33, 0.33, false);//0.33
		lstm_build1.init(&(modelparams->lstm_param1), 0.33, 0.33, true);//0.33
		lstm_build_right1.init(&(modelparams->lstm_param_right1), 0.33, 0.33, false);//0.33
		lstm_build2.init(&(modelparams->lstm_param2), 0.33, 0.33, true);//0.33
		lstm_build_right2.init(&(modelparams->lstm_param_right2), 0.33, 0.33, false);//0.33
	}

	void multi_initial() {
		for (int idx = 0; idx < length; idx++) {
			word_pret_emb[idx].init(hyperparams->word_emb_size, -1);
			word_rand_emb[idx].init(hyperparams->word_emb_size, -1);
			word_emb[idx].init(hyperparams->word_emb_size, -1);
			word_emb_drop[idx].init(hyperparams->word_emb_size, 0.33);
			tag_emb[idx].init(hyperparams->tag_emb_size, -1);
			tag_emb_drop[idx].init(hyperparams->tag_emb_size, 0.33);
			emb_input[idx].init(hyperparams->word_emb_size + hyperparams->tag_emb_size, -1);

			lstm_output[idx].init(hyperparams->lstm_output_size * 2, -1);
			lstm_output1[idx].init(hyperparams->lstm_output_size * 2, -1);
			lstm_output2[idx].init(hyperparams->lstm_output_size * 2, -1);//0.33
			lstm_drop[idx].init(hyperparams->lstm_output_size * 2, 0.33);

			mlp_head[idx].init(hyperparams->mlp_size, -1);
			mlp_head[idx].setFunctions(fleaky_relu, dleaky_relu);
			mlp_head_drop[idx].init(hyperparams->mlp_size, 0.33);

			mlp_dep[idx].init(hyperparams->mlp_size, -1);
			mlp_dep[idx].setFunctions(fleaky_relu, dleaky_relu);
			mlp_dep_drop[idx].init(hyperparams->mlp_size, 0.33);

			mlp_head_rel[idx].init(hyperparams->mlp_rel_size, -1);
			mlp_head_rel[idx].setFunctions(fleaky_relu, dleaky_relu);
			mlp_head_rel_drop[idx].init(hyperparams->mlp_rel_size, 0.33);

			mlp_dep_rel[idx].init(hyperparams->mlp_rel_size, -1);
			mlp_dep_rel[idx].setFunctions(fleaky_relu, dleaky_relu);
			mlp_dep_rel_drop[idx].init(hyperparams->mlp_rel_size, 0.33);
		}

		arc_biaffine.init(length);
		rel_biaffine.init(length);
	}

	void multi_forward(Graph *pcg, dparser::Instance* inst) {
		if (pcg->train) {
			multi_get_drop_mask();
		}

		for (int i = 0; i < length; i++) {
			word_pret_emb[i].forward(pcg, inst->pret_forms_id[i]);
			word_rand_emb[i].forward(pcg, inst->forms_id[i]);
			word_emb[i].forward(pcg, &word_pret_emb[i], &word_rand_emb[i]);
			word_emb_drop[i].forward(pcg, &word_emb[i]);
			tag_emb[i].forward(pcg, inst->pos_id[i]);
			tag_emb_drop[i].forward(pcg, &tag_emb[i]);
			emb_input[i].forward(pcg, &word_emb_drop[i], &tag_emb_drop[i]);
		}

		lstm_build.forward(pcg, getPNodes(emb_input, length));
		lstm_build_right.forward(pcg, getPNodes(emb_input, length));

		for (int i = 0; i < length; i++) {
			lstm_output[i].forward(pcg, &lstm_build._hiddens[i], &lstm_build_right._hiddens[i]);
		}

		lstm_build1.forward(pcg, getPNodes(lstm_output, length));
		lstm_build_right1.forward(pcg, getPNodes(lstm_output, length));

		for (int i = 0; i < length; i++) {
			lstm_output1[i].forward(pcg, &lstm_build1._hiddens[i], &lstm_build_right1._hiddens[i]);
		}

		for (int i = 0; i < length; i++) {
			lstm_drop[i].forward(pcg, &lstm_output1[i]);
		}
		

		for (int i = 0; i < length; i++) {
			mlp_head[i].forward(pcg, &lstm_drop[i]);
			mlp_dep[i].forward(pcg, &lstm_drop[i]);
			mlp_head_drop[i].forward(pcg, &mlp_head[i]);
			mlp_dep_drop[i].forward(pcg, &mlp_dep[i]);
		}


		arc_biaffine.forward(pcg, getPNodes(mlp_dep_drop, length), getPNodes(mlp_head_drop, length));

		for (int i = 0; i < length; i++) {
			mlp_head_rel[i].forward(pcg, &lstm_drop[i]);
			mlp_dep_rel[i].forward(pcg, &lstm_drop[i]);
			mlp_head_rel_drop[i].forward(pcg, &mlp_head_rel[i]);
			mlp_dep_rel_drop[i].forward(pcg, &mlp_dep_rel[i]);
		}

		rel_biaffine.forward(pcg, getPNodes(mlp_dep_rel_drop, length), getPNodes(mlp_head_rel_drop, length));
	}
	// conversion
	void conversion_resize(int len) {
		length = len;

		word_pret_emb.resize(len);
		word_rand_emb.resize(len);
		word_emb.resize(len);
		word_emb_drop.resize(len);
		tag_emb.resize(len);
		tag_emb_drop.resize(len);


		emb_input.resize(len);

		lcn_label_emb.resize(len);
		h_label_emb.resize(len);
		m_label_emb.resize(len);


		label_emb.resize(len);

		lstm_build.resize(len);
		lstm_build_right.resize(len);
		lstm_output.resize(len);


		lstm_build1.resize(len);
		lstm_build_right1.resize(len);
		lstm_output1.resize(len);

		lstm_drop.resize(len);


		sp_input.resize(len);

		tree_lstm_build.resize(len);
		for (int i = 0; i < len; i++) {
			tree_lstm_build[i].resize(len);
			for (int j = 0; j < len; j++) {
				tree_lstm_build[i][j].resize(len);
			}
		}
		tree_lstm_build_top_down.resize(len);
		for (int i = 0; i < len; i++) {
			tree_lstm_build_top_down[i].resize(len);
			for (int j = 0; j < len; j++) {
				tree_lstm_build_top_down[i][j].resize(len);
			}
		}

		expand_pattern_emb.resize(len);
		for (int i = 0; i < len; i++) {
			expand_pattern_emb[i].resize(len);
		}

		conv_mlp_head.resize(len);
		conv_mlp_head_drop.resize(len);
		for (int i = 0; i < len; i++) {
			conv_mlp_head[i].resize(len);
			conv_mlp_head_drop[i].resize(len);
		}
		conv_mlp_dep.resize(len);
		conv_mlp_dep_drop.resize(len);
		for (int i = 0; i < len; i++) {
			conv_mlp_dep[i].resize(len);
			conv_mlp_dep_drop[i].resize(len);
		}

		conv_mlp_head_rel.resize(len);
		conv_mlp_head_rel_drop.resize(len);
		for (int i = 0; i < len; i++) {
			conv_mlp_head_rel[i].resize(len);
			conv_mlp_head_rel_drop[i].resize(len);
		}

		conv_mlp_dep_rel.resize(len);
		conv_mlp_dep_rel_drop.resize(len);
		for (int i = 0; i < len; i++) {
			conv_mlp_dep_rel[i].resize(len);
			conv_mlp_dep_rel_drop[i].resize(len);
		}

		concat_sp.resize(len);
		for (int i = 0; i < len; i++) {
			concat_sp[i].resize(len);
		}

		concat_head.resize(len);
		for (int i = 0; i < len; i++) {
			concat_head[i].resize(len);
		}

		concat_dep.resize(len);
		for (int i = 0; i < len; i++) {
			concat_dep[i].resize(len);
		}

		arc_biaffines.resize(len);
		for (int i = 0; i < len; i++) {
			arc_biaffines[i].resize(len);
		}

		rel_biaffines.resize(len);
		for (int i = 0; i < len; i++) {
			rel_biaffines[i].resize(len);
			for (int j = 0; j < len; j++) {
				rel_biaffines[i][j].resize(modelparams->sbiaffine_rel.size());
			}
		}

		biaffine_inputs_dep.resize(len);
		for (int i = 0; i < len; i++) { biaffine_inputs_dep[i].resize(len); }

		biaffine_inputs_head.resize(len);
		for (int i = 0; i < len; i++) { biaffine_inputs_head[i].resize(len); }
		
		biaffine_inputs_dep_rel.resize(len);
		for (int i = 0; i < len; i++) { biaffine_inputs_dep_rel[i].resize(len); }
		
		biaffine_inputs_head_rel.resize(len);
		for (int i = 0; i < len; i++) { biaffine_inputs_head_rel[i].resize(len); }
	}

	void conversion_setParams() {
		for (int idx = 0; idx < length; idx++) {
			word_pret_emb[idx].setParam(&(modelparams->pret_word_embs));
			word_rand_emb[idx].setParam(&(modelparams->word_embs));
			tag_emb[idx].setParam(&(modelparams->tag_embs));

			lcn_label_emb[idx].setParam(&(modelparams->lcn_label_emb));
			m_label_emb[idx].setParam(&(modelparams->m_label_emb));
			h_label_emb[idx].setParam(&(modelparams->h_label_emb));

			label_emb[idx].setParam(&(modelparams->label_emb));

			for (int idy = 0; idy < length; idy++) {
				expand_pattern_emb[idx][idy].setParam(&(modelparams->expand_pattern_param));

				conv_mlp_head[idx][idy].setParam(&(modelparams->conv_mlp_head_param));
				conv_mlp_dep[idx][idy].setParam(&(modelparams->conv_mlp_dep_param));
				conv_mlp_head_rel[idx][idy].setParam(&(modelparams->conv_mlp_head_rel_param));
				conv_mlp_dep_rel[idx][idy].setParam(&(modelparams->conv_mlp_dep_rel_param));

				for (int idz = 0; idz < modelparams->sbiaffine_rel.size(); idz++) {
					rel_biaffines[idx][idy][idz].setParam(&(modelparams->sbiaffine_rel[idz]), 2, 2);
				}
				arc_biaffines[idx][idy].setParam(&modelparams->sbiaffine_arc, 1, 0);
				tree_lstm_build[idx][idy].init(&(modelparams->tree_lstm_param), -1, true);
				tree_lstm_build_top_down[idx][idy].init(&(modelparams->tree_lstm_param_top_down), -1, false);
			}

		}

		lstm_build.init(&(modelparams->lstm_param), 0.33, 0.33, true);//0.33
		lstm_build_right.init(&(modelparams->lstm_param_right), 0.33, 0.33, false);//0.33
		lstm_build1.init(&(modelparams->lstm_param1), 0.33, 0.33, true);//0.33
		lstm_build_right1.init(&(modelparams->lstm_param_right1), 0.33, 0.33, false);//0.33
	}

	void conversion_initial() {
		for (int idx = 0; idx < length; idx++) {
			word_pret_emb[idx].init(hyperparams->word_emb_size, -1);
			word_rand_emb[idx].init(hyperparams->word_emb_size, -1);
			word_emb[idx].init(hyperparams->word_emb_size, -1);
			word_emb_drop[idx].init(hyperparams->word_emb_size, 0.33);
			tag_emb[idx].init(hyperparams->tag_emb_size, -1);
			tag_emb_drop[idx].init(hyperparams->tag_emb_size, 0.33);
			emb_input[idx].init(hyperparams->word_emb_size + hyperparams->tag_emb_size, -1);


			lcn_label_emb[idx].init(hyperparams->lcn_label_emb_size, -1);
			m_label_emb[idx].init(hyperparams->m_label_emb_size, -1);
			h_label_emb[idx].init(hyperparams->h_label_emb_size, -1);

			label_emb[idx].init(hyperparams->label_emb_size, -1);
			sp_input[idx].init(hyperparams->label_emb_size + hyperparams->lstm_output_size * 2, -1);

			lstm_output[idx].init(hyperparams->lstm_output_size * 2, -1);
			lstm_output1[idx].init(hyperparams->lstm_output_size * 2, -1);
			lstm_drop[idx].init(hyperparams->lstm_output_size * 2, 0.33);

			for (int idy = 0; idy < length; idy++) {
				expand_pattern_emb[idx][idy].init(hyperparams->pattern_emb_size, -1);

				conv_mlp_head[idx][idy].init(hyperparams->mlp_size, -1);
				conv_mlp_head[idx][idy].setFunctions(fleaky_relu, dleaky_relu);
				conv_mlp_head_drop[idx][idy].init(hyperparams->mlp_size, 0.33);

				conv_mlp_dep[idx][idy].init(hyperparams->mlp_size, -1);
				conv_mlp_dep[idx][idy].setFunctions(fleaky_relu, dleaky_relu);
				conv_mlp_dep_drop[idx][idy].init(hyperparams->mlp_size, 0.33);

				conv_mlp_head_rel[idx][idy].init(hyperparams->mlp_rel_size, -1);
				conv_mlp_head_rel[idx][idy].setFunctions(fleaky_relu, dleaky_relu);
				conv_mlp_head_rel_drop[idx][idy].init(hyperparams->mlp_rel_size, 0.33);

				conv_mlp_dep_rel[idx][idy].init(hyperparams->mlp_rel_size, -1);
				conv_mlp_dep_rel[idx][idy].setFunctions(fleaky_relu, dleaky_relu);
				conv_mlp_dep_rel_drop[idx][idy].init(hyperparams->mlp_rel_size, 0.33);

				concat_sp[idx][idy].init(hyperparams->tree_lstm_output_size * 3, -1);
				concat_head[idx][idy].init(hyperparams->tree_lstm_output_size * 3 + hyperparams->lstm_output_size * 2 +
					hyperparams->pattern_emb_size + hyperparams->m_label_emb_size + hyperparams->h_label_emb_size + hyperparams->lcn_label_emb_size, -1);
				concat_dep[idx][idy].init(hyperparams->tree_lstm_output_size * 3 + hyperparams->lstm_output_size * 2 +
					hyperparams->pattern_emb_size + hyperparams->m_label_emb_size + hyperparams->h_label_emb_size + hyperparams->lcn_label_emb_size, -1);

				arc_biaffines[idx][idy].init(1, -1);
				for (int idz = 0; idz < modelparams->sbiaffine_rel.size(); idz++) {
					rel_biaffines[idx][idy][idz].init(1, -1);
				}
			}
		}
	}

	void conversion_forward(Graph *pcg, dparser::Instance* inst) {
		if (pcg->train) {
			sp_label_conv_get_drop_mask();
		}

		vector<TreeNode> tree(length);
		for (int i = 0; i < length; i++) {
			tree[i].parent = inst->heads2[i];
			if (inst->heads2[i] >= 0) {
				tree[inst->heads2[i]].children.push_back(i);
			}
		}
		for (int i = 0; i < length; i++) {
			tree[i].degree = tree[i].children.size();
		}

		for (int i = 0; i < length; i++) {
			word_pret_emb[i].forward(pcg, inst->pret_forms_id[i]);
			word_rand_emb[i].forward(pcg, inst->forms_id[i]);
			word_emb[i].forward(pcg, &word_pret_emb[i], &word_rand_emb[i]);
			word_emb_drop[i].forward(pcg, &word_emb[i]);
			tag_emb[i].forward(pcg, inst->pos_id[i]);
			tag_emb_drop[i].forward(pcg, &tag_emb[i]);
			emb_input[i].forward(pcg, &word_emb_drop[i], &tag_emb_drop[i]);
		}

		lstm_build.forward(pcg, getPNodes(emb_input, length));
		lstm_build_right.forward(pcg, getPNodes(emb_input, length));

		for (int i = 0; i < length; i++) {
			lstm_output[i].forward(pcg, &lstm_build._hiddens[i], &lstm_build_right._hiddens[i]);
		}

		lstm_build1.forward(pcg, getPNodes(lstm_output, length));
		lstm_build_right1.forward(pcg, getPNodes(lstm_output, length));

		for (int i = 0; i < length; i++) {
			lstm_output1[i].forward(pcg, &lstm_build1._hiddens[i], &lstm_build_right1._hiddens[i]);
		}

		for (int i = 0; i < length; i++) {
			lstm_drop[i].forward(pcg, &lstm_output1[i]);
			label_emb[i].forward(pcg, inst->labels_id2[i]);
			sp_input[i].forward(pcg, &lstm_drop[i], &label_emb[i]);
		}


		for (int i = 0; i < length; i++) {
			m_label_emb[i].forward(pcg, inst->labels_id2[i]);
			h_label_emb[i].forward(pcg, inst->labels_id2[i]);
			lcn_label_emb[i].forward(pcg, inst->labels_id2[i]);
		}

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				int pindex = which_expand_pattern(i, j, inst->heads2);
				expand_pattern_emb[i][j].forward(pcg, pindex);
			}
		}

		//sp-tree
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				int lcn_bu;
				tree_lstm_build[i][j].forward(pcg, getPNodes(sp_input, length), tree, i, j, lcn_bu);
				int lcn_td;
				tree_lstm_build_top_down[i][j].forward(pcg, getPNodes(sp_input, length), tree, i, j, lcn_td);

				concat_sp[i][j].forward(pcg, &tree_lstm_build_top_down[i][j]._hiddens[i], &tree_lstm_build[i][j]._hiddens[lcn_bu], &tree_lstm_build_top_down[i][j]._hiddens[j]);

				concat_head[i][j].forward(pcg, &lstm_drop[j], &concat_sp[i][j], &expand_pattern_emb[i][j], &m_label_emb[i], &lcn_label_emb[lcn_bu], &h_label_emb[j]);
				concat_dep[i][j].forward(pcg, &lstm_drop[i], &concat_sp[i][j], &expand_pattern_emb[i][j], &m_label_emb[i], &lcn_label_emb[lcn_bu], &h_label_emb[j]);
			}
		}




		//@zhangxinzhou zhangbo
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				conv_mlp_head[i][j].forward(pcg, &concat_head[i][j]);
				conv_mlp_dep[i][j].forward(pcg, &concat_dep[i][j]);
				conv_mlp_head_drop[i][j].forward(pcg, &conv_mlp_head[i][j]);
				conv_mlp_dep_drop[i][j].forward(pcg, &conv_mlp_dep[i][j]);
			}
		}

		for (int i = 0; i < length; i++) {
			for (int j = 0; j < length; j++) {
				arc_biaffines[i][j].forward(pcg, &conv_mlp_dep_drop[i][j], &conv_mlp_head_drop[i][j]);
			}
		}

		//arc_biaffine.forward(pcg, getPNodes(mlp_dep, len), getPNodes(mlp_head, len));
		int label_num = modelparams->sbiaffine_rel.size();
		if (pcg->train) {
			for (int i = 0; i < length; i++) {
				int h = inst->heads[i];
				if (h == -1) continue;
				conv_mlp_head_rel[i][h].forward(pcg, &concat_head[i][h]);
				conv_mlp_dep_rel[i][h].forward(pcg, &concat_dep[i][h]);
				conv_mlp_head_rel_drop[i][h].forward(pcg, &conv_mlp_head_rel[i][h]);
				conv_mlp_dep_rel_drop[i][h].forward(pcg, &conv_mlp_dep_rel[i][h]);

				for (int k = 0; k < label_num; k++) {
					rel_biaffines[i][h][k].forward(pcg, &conv_mlp_dep_rel_drop[i][h], &conv_mlp_head_rel_drop[i][h]);
				}
			}
		}
		else {
			for (int i = 0; i < length; i++) {
				for (int j = 0; j < length; j++) {
					conv_mlp_head_rel[i][j].forward(pcg, &concat_head[i][j]);
					conv_mlp_dep_rel[i][j].forward(pcg, &concat_dep[i][j]);
					conv_mlp_head_rel_drop[i][j].forward(pcg, &conv_mlp_head_rel[i][j]);
					conv_mlp_dep_rel_drop[i][j].forward(pcg, &conv_mlp_dep_rel[i][j]);

					for (int k = 0; k < label_num; k++) {
						rel_biaffines[i][j][k].forward(pcg, &conv_mlp_dep_rel_drop[i][j], &conv_mlp_head_rel_drop[i][j]);
					}
				}
			}
		}
	}

	void forward(Graph *pcg, dparser::Instance* inst) {
		if (inst->type.compare("multi-source") == 0)
		{
			clear();
			resize(inst->size());
			multi_source_setParams();
			multi_initial();
			multi_forward(pcg, inst);
		}
		else if (inst->type.compare("multi-target") == 0)
		{
			clear();
			resize(inst->size());
			multi_target_setParams();
			multi_initial();
			multi_forward(pcg, inst);
		}
		else if (inst->type.compare("conversion") == 0)
		{
			clear();
			conversion_resize(inst->size());
			conversion_setParams();
			conversion_initial();
			conversion_forward(pcg, inst);
		}
		else
		{
			assert(false);
		}
	}

	void multi_get_drop_mask() {
		get_emb_drop_mask();

		Tensor1D lstm_mask;
		lstm_mask.init(lstm_output1[0].dim);
		getMask(lstm_mask, 0.33);


		Tensor1D mlp_dep_mask;
		mlp_dep_mask.init(mlp_dep[0].dim);
		getMask(mlp_dep_mask, 0.33);

		Tensor1D mlp_head_mask;
		mlp_head_mask.init(mlp_head[0].dim);
		getMask(mlp_head_mask, 0.33);

		Tensor1D mlp_dep_rel_mask;
		mlp_dep_rel_mask.init(mlp_dep_rel[0].dim);
		getMask(mlp_dep_rel_mask, 0.33);


		Tensor1D mlp_head_rel_mask;
		mlp_head_rel_mask.init(mlp_head_rel[0].dim);
		getMask(mlp_head_rel_mask, 0.33);


		for (int i = 0; i < length; i++) {
			lstm_drop[i].setMask(lstm_mask);
			mlp_dep_drop[i].setMask(mlp_dep_mask);
			mlp_head_drop[i].setMask(mlp_head_mask);
			mlp_dep_rel_drop[i].setMask(mlp_dep_rel_mask);
			mlp_head_rel_drop[i].setMask(mlp_head_rel_mask);
		}
	}

	void synchronization_get_drop_mask() {
		get_emb_drop_mask();

		Tensor1D lstm_mask;
		lstm_mask.init(lstm_output1[0].dim);
		getMask(lstm_mask, 0.33);


		Tensor1D mlp_dep_mask;
		mlp_dep_mask.init(mlp_dep[0].dim);
		getMask(mlp_dep_mask, 0.33);

		Tensor1D mlp_head_mask;
		mlp_head_mask.init(mlp_head[0].dim);
		getMask(mlp_head_mask, 0.33);

		Tensor1D mlp_dep_rel_mask;
		mlp_dep_rel_mask.init(mlp_dep_rel[0].dim);
		getMask(mlp_dep_rel_mask, 0.33);


		Tensor1D mlp_head_rel_mask;
		mlp_head_rel_mask.init(mlp_head_rel[0].dim);
		getMask(mlp_head_rel_mask, 0.33);


		for (int i = 0; i < length; i++) {
			lstm_drop[i].setMask(lstm_mask);
			mlp_dep_drop[i].setMask(mlp_dep_mask);
			mlp_head_drop[i].setMask(mlp_head_mask);
			mlp_dep_rel_drop[i].setMask(mlp_dep_rel_mask);
			mlp_head_rel_drop[i].setMask(mlp_head_rel_mask);

			mlp_dep_drop2[i].setMask(mlp_dep_mask);
			mlp_head_drop2[i].setMask(mlp_head_mask);
			mlp_dep_rel_drop2[i].setMask(mlp_dep_rel_mask);
			mlp_head_rel_drop2[i].setMask(mlp_head_rel_mask);
		}
	}

	void sp_label_conv_get_drop_mask() {
		get_emb_drop_mask();

		Tensor1D lstm_mask;
		lstm_mask.init(lstm_output1[0].dim);
		getMask(lstm_mask, 0.33);


		Tensor1D mlp_dep_mask;
		mlp_dep_mask.init(conv_mlp_dep[0][0].dim);
		getMask(mlp_dep_mask, 0.33);

		Tensor1D mlp_head_mask;
		mlp_head_mask.init(conv_mlp_head[0][0].dim);
		getMask(mlp_head_mask, 0.33);

		Tensor1D mlp_dep_rel_mask;
		mlp_dep_rel_mask.init(conv_mlp_dep_rel[0][0].dim);
		getMask(mlp_dep_rel_mask, 0.33);


		Tensor1D mlp_head_rel_mask;
		mlp_head_rel_mask.init(conv_mlp_head_rel[0][0].dim);
		getMask(mlp_head_rel_mask, 0.33);


		for (int i = 0; i < length; i++) {
			lstm_drop[i].setMask(lstm_mask);
			for (int j = 0; j < length; j++) {
				conv_mlp_dep_drop[i][j].setMask(mlp_dep_mask);
				conv_mlp_head_drop[i][j].setMask(mlp_head_mask);

				conv_mlp_dep_rel_drop[i][j].setMask(mlp_dep_rel_mask);
				conv_mlp_head_rel_drop[i][j].setMask(mlp_head_rel_mask);
			}
		}
	}

	void tree_convert_get_drop_mask() {
		get_emb_drop_mask();

		Tensor1D lstm_mask;
		lstm_mask.init(lstm_output1[0].dim);
		getMask(lstm_mask, 0.33);


		Tensor1D mlp_dep_mask;
		mlp_dep_mask.init(conv_mlp_dep[0][0].dim);
		getMask(mlp_dep_mask, 0.33);

		Tensor1D mlp_head_mask;
		mlp_head_mask.init(conv_mlp_head[0][0].dim);
		getMask(mlp_head_mask, 0.33);

		Tensor1D mlp_dep_rel_mask;
		mlp_dep_rel_mask.init(mlp_dep_rel[0].dim);
		getMask(mlp_dep_rel_mask, 0.33);


		Tensor1D mlp_head_rel_mask;
		mlp_head_rel_mask.init(mlp_head_rel[0].dim);
		getMask(mlp_head_rel_mask, 0.33);


		for (int i = 0; i < length; i++) {
			lstm_drop[i].setMask(lstm_mask);
			for (int j = 0; j < length; j++) {
				conv_mlp_dep_drop[i][j].setMask(mlp_dep_mask);
				conv_mlp_head_drop[i][j].setMask(mlp_head_mask);
			}
			/*mlp_dep_drop[i].setMask(mlp_dep_mask);
			mlp_head_drop[i].setMask(mlp_head_mask);*/
			mlp_dep_rel_drop[i].setMask(mlp_dep_rel_mask);
			mlp_head_rel_drop[i].setMask(mlp_head_rel_mask);
		}
	}

	void get_emb_drop_mask() {
		for (int i = 0; i < length; i++) {
			Tensor1D wMask;
			wMask.init(word_emb[i].dim);
			wMask = 1;

			Tensor1D tMask;
			tMask.init(tag_emb[i].dim);
			tMask = 1;

			getEmbMask(wMask, tMask, 0.33);
			word_emb_drop[i].setMask(wMask);
			tag_emb_drop[i].setMask(tMask);
		}
	}

	bool isPatten1(int m, int h, const vector<int> &heads2) {
			if (h == heads2[m]) {
				return true;
			}
			return false;
		}

	bool isPatten2(int m, int h, const vector<int> &heads2) {
			if (h == heads2[heads2[m]]) {
				return true;
			}
			return false;
		}

	bool isPatten3(int m, int h, const vector<int> &heads2) {
			if (heads2[h] == heads2[m]) {
				return true;
			}
			return false;
		}

	bool isPatten4(int m, int h, const vector<int> &heads2) {
			if (m == heads2[h]) {
				return true;
			}
			return false;
		}

	bool isPatten5(int m, int h, const vector<int> &heads2) {
			if (m == heads2[heads2[h]]) {
				return true;
			}
			return false;
		}

	int which_pattern(int m, int h, const vector<int> &heads2) {
			if (isPatten1(m, h, heads2)) return 0; //"consistent";
			if (isPatten2(m, h, heads2)) return 1; //"grand";
			if (isPatten3(m, h, heads2)) return 2; //"sibling";
			if (isPatten4(m, h, heads2)) return 3; //"reverse";
			if (isPatten5(m, h, heads2)) return 4; //"reverse-grand"
			return 5;
		}

	int which_expand_pattern(int m, int h, const vector<int> &heads2) {
			if (isPatten1(m, h, heads2)) return 0; //"consistent";
			if (isPatten2(m, h, heads2)) return 1; //"grand";
			if (isPatten3(m, h, heads2)) return 2; //"sibling";
			if (isPatten4(m, h, heads2)) return 3; //"reverse";
			if (isPatten5(m, h, heads2)) return 4; //"reverse-grand";

			int dist = get_dist(m, h, heads2);
			int i;
			for (i = 1; i < hyperparams->range.size(); i++) {
				if (dist >= hyperparams->range[i - 1] && dist < hyperparams->range[i]) {
					return i + 4;
				}
			}
			assert(dist >= hyperparams->range[hyperparams->range.size() - 1]);
			return i + 4;
		}

	int get_dist(int m, int h, const vector<int>& heads2) {
		vector<int> mparents;
		mparents.push_back(m);
		while (m != -1) {
			mparents.push_back(heads2[m]);
			m = heads2[m];
		}

		vector<int> hparents;
		hparents.push_back(h);
		while (h != -1) {
			hparents.push_back(heads2[h]);
			h = heads2[h];
		}

		for (int i = 0; i < mparents.size(); i++) {
			for (int j = 0; j < hparents.size(); j++) {
				if (mparents[i] == hparents[j]) {
					assert(i + j >  0 || i == j);
					return i + j;
				}
			}
		}
	}

	int get_lcn(int m, int h, const vector<int>& heads2) {
		vector<int> mparents;
		mparents.push_back(m);
		while (m != -1) {
			mparents.push_back(heads2[m]);
			m = heads2[m];
		}

		vector<int> hparents;
		hparents.push_back(h);
		while (h != -1) {
			hparents.push_back(heads2[h]);
			h = heads2[h];
		}

		for (int i = 0; i < mparents.size(); i++) {
			for (int j = 0; j < hparents.size(); j++) {
				if (mparents[i] == hparents[j]) {
					assert(i + j >  0 || i == j);
					return mparents[i];
				}
			}
		}
	}
};

#endif
