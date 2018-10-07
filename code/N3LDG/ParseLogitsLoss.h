#ifndef  _PARSELOGITSLOSS_H_
#define  _PARSELOGITSLOSS_H_

#include "BaseLoss.h"
#include "Biaffine.h"
#include "Instance.h"

namespace dparser {

	class ParseLogitsLoss : public BaseLoss {
	public:
		inline dtype loss(vector<vector<SBiaffineNode>> &x,  dparser::Instance* inst, int word_num) {
			int len = inst->size();
			vector<int> heads = (inst->type.compare("multi-source") == 0) ? inst->heads2 : inst->heads;

			vector<vector<double> > head_prob(len);
			for (int i = 0; i < len; i++) {
				head_prob[i].resize(len);
				for (int j = 0; j < len; j++) {
					head_prob[i][j] = x[i][j].val.v[0];
				}
			}

			for (int i = 0; i < len; i++) {
				if (i == 0) {
					continue;
				}
				if (heads[i] == -1) {
					continue;
				}
				double Z = 0.0;
				double max_score = -10e20;
				for (int j = 0; j < len; j++) {
					if (head_prob[i][j] > max_score)
						max_score = head_prob[i][j];
				}
				for (int j = 0; j < len; j++) {
					Z += exp(head_prob[i][j] - max_score);
				}
				for (int j = 0; j < len; j++) {
					double loss = exp(head_prob[i][j] - max_score) / Z + (heads[i] == j ? -1.0 : 0.0);
					// parse_logits.losses[0].mat()(i, j) += loss;
					//x->losses[0].assign(j, i, x->losses[0].get(j, i) + loss);
					x[i][j].loss.v[0] += loss / word_num;
				}
			}
		}


		inline dtype loss(BiaffineNode* x, dparser::Instance* inst, int word_num) {
			int len = inst->size();
			vector<int> heads = (inst->type.compare("multi-source") == 0) ? inst->heads2 : inst->heads;

			vector<vector<double> > head_prob(len);
			for (int i = 0; i < len; i++) {
				head_prob[i].resize(len);
				for (int j = 0; j < len; j++) {
					head_prob[i][j] = x->vals[0].mat()(i, j);
				}
			}

			for (int i = 0; i < len; i++) {
				if (i == 0) {
					continue;
				}
				if (heads[i] == -1) {
					for (int j = 0; j < len; j++) {
						x->losses[0].mat()(i, j) = 0;
					}
					continue;
				}
				double Z = 0.0;
				double max_score = -10e20;
				for (int j = 0; j < len; j++) {
					if (head_prob[i][j] > max_score)
						max_score = head_prob[i][j];
				}
				for (int j = 0; j < len; j++) {
					Z += exp(head_prob[i][j] - max_score);
				}
				for (int j = 0; j < len; j++) {
					double loss = exp(head_prob[i][j] - max_score) / Z + (heads[i] == j ? -1.0 : 0.0);
					// parse_logits.losses[0].mat()(i, j) += loss;
				   //x->losses[0].assign(j, i, x->losses[0].get(j, i) + loss);
					x->losses[0].mat()(i, j) += loss / word_num;
				}
			}
		}

		inline dtype rel_loss(BiaffineNode* x, const dparser::Instance* inst, int word_num) {
			int len = inst->size();
			vector<int> heads = (inst->type.compare("multi-source") == 0) ? inst->heads2 : inst->heads;
			vector<int> labels_id = (inst->type.compare("multi-source") == 0) ? inst->labels_id2 : inst->labels_id;


			int label_num = x->classDim;
			vector<vector<vector<dtype>>> rel_prob;
			rel_prob.resize(len);
			for (int i = 0; i < len; i++) {
				rel_prob[i].resize(len);
				for (int j = 0; j < len; j++) {
					rel_prob[i][j].resize(label_num);
				}
			}

			for (int i = 0; i < len; i++) {
				for (int j = 0; j < len; j++) {
					for (int r = 0; r < label_num; r++) {
						rel_prob[i][j][r] = x->vals[r].mat()(i, j);
					}
				}
			}

			for (int i = 1; i < len; i++) {
				if (labels_id[i] == -1) {
					continue;
				}

				int head = heads[i];

				double Z = 0.0;
				double max_score = -10e20;

				for (int r = 0; r < label_num; r++) {
					if (rel_prob[i][head][r] > max_score) {
						max_score = rel_prob[i][head][r];
					}
				}

				for (int r = 0; r < label_num; r++) {
					Z += exp(rel_prob[i][head][r] - max_score);
				}

				for (int r = 0; r < label_num; r++) {
					dtype loss = exp(rel_prob[i][head][r] - max_score) / Z + ((labels_id[i] - 2 == r) ? -1.0 : 0.0);
					x->losses[r].mat()(i, head) += loss / word_num;
				}
			}
		}

		inline dtype rel_loss(vector<vector<vector<SBiaffineNode>>> &x, dparser::Instance* inst, int word_num) { assert(0); return 0; }
		inline void constrained_predict(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) {
			assert(0);
		}
		inline dtype rel_loss(BiaffineNode* x, const dparser::Instance* inst, int word_num, int which) { assert(0); return 0; }
		inline dtype loss(BiaffineNode *x, dparser::Instance* inst, int word_num, int which) { assert(0); return 0; } 

		inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, BiaffineNode* rel, dparser::Instance* inst) {
			int len = inst->predicted_heads.size();
			int label_num = rel->classDim;
			vector<vector<double> > head_prob(len);
			for (int i = 0; i < len; i++) {
				head_prob[i].resize(len);
				for (int j = 0; j < len; j++) {
					head_prob[i][j] = arc[i][j].val.v[0];;
				}
			}

			for (int i = 0; i < len; i++) {
				double max = head_prob[i][0];
				int index = 0;
				for (int j = 0; j < len; j++) {
					if (head_prob[i][j] > max + 1e-20) {
						index = j;
						max = head_prob[i][j];
					}
				}
				inst->predicted_heads[i] = index;
			}

			for (int i = 1; i < len; i++) {
				int predict_head = inst->predicted_heads[i];
				int max_index = 0;
				dtype max_prob = rel->vals[0].mat()(i, predict_head);
				for (int r = 1; r < label_num; r++) {
					if (rel->vals[r].mat()(i, predict_head) > max_prob) {
						max_index = r;
						max_prob = rel->vals[r].mat()(i, predict_head);
					}
				}

				inst->predicted_labels_id[i] = max_index + 2;
			}

		}


		inline dtype predict_arc_and_rel(BiaffineNode* arc, BiaffineNode* rel, dparser::Instance* inst) {
			int len = inst->predicted_heads.size();
			int label_num = rel->classDim;
			vector<vector<double> > head_prob(len);
			for (int i = 0; i < len; i++) {
				head_prob[i].resize(len);
				for (int j = 0; j < len; j++) {
					head_prob[i][j] = arc->vals[0].mat()(i, j);
				}
			}

			for (int i = 0; i < len; i++) {
				double max = head_prob[i][0];
				int index = 0;
				for (int j = 0; j < len; j++) {
					if (head_prob[i][j] > max + 1e-20) {
						index = j;
						max = head_prob[i][j];
					}
				}
				inst->predicted_heads[i] = index;
			}

			for (int i = 1; i < len; i++) {
				int predict_head = inst->predicted_heads[i];
				int max_index = 0;
				dtype max_prob = rel->vals[0].mat()(i, predict_head);
				for (int r = 1; r < label_num; r++) {
					if (rel->vals[r].mat()(i, predict_head) > max_prob) {
						max_index = r;
						max_prob = rel->vals[r].mat()(i, predict_head);
					}
				}

				inst->predicted_labels_id[i] = max_index + 2;
			}

		}

		inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) { assert(0); return 0; }
	};


}

#endif
