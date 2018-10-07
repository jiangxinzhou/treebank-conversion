#ifndef _INSTANCE_
#define _INSTANCE_

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "CharUtils.h"
#include "Parameters.h"
#include "FVec.h"
#include "NRMat.h"
using namespace nr;
using namespace std;
using namespace egstra;

namespace dparser {

	class Instance
	{
	public:
		Instance(const int _id) : id(_id) {}
		~Instance() {}
		int size() const { return forms.size(); }

	public:
		//@jiangxinzhou zhangbo
		vector<int> heads2;
		vector<string> deprels2;
		//jxz
		string type;
		vector<int> labels_id2;

		const int id;

		//for crf-loss
		NRMat3d< double > prob_dep;
		NRMat< double > marg_prob_base; //[head-idx][mod-idx]
		NRMat< double > marg_prob_answer;
		const NRMat<bool> * p_cand_heads;

		// 总是直接调用
		//    set_candidate_heads_max(candidate_heads_base)
		NRMat<bool> candidate_heads_base; // 整个搜索空间：n*n，[mod-idx][head-idx]，true表示可以有弧，false表示不可以有弧
		NRMat<bool> candidate_heads_answer; // 正确答案搜索空间，如果数据为完整标注数据，即一棵树，那么对应只有n个true，每个词（行向量）一个true

		void set_candidate_heads_max(NRMat<bool> &candidate_heads) {
			const int len = size();
			candidate_heads.resize(len, len);
			candidate_heads = true;
			for (int m = 0; m < size(); ++m) {
				candidate_heads[0][m] = false;
				candidate_heads[m][m] = false;
			}
		}

		void set_candidate_heads_single_head_for_one_word(NRMat<bool> &candidate_heads, int m, int h) {
			assert(m > 0 && m < size());
			assert(h >= 0 && h < size());
			for (int i = 0; i < size(); ++i) {
				candidate_heads[m][i] = false;
			}
			candidate_heads[m][h] = true;
		}

		bool need_completion() {
			for (int i = 1; i < this->size(); i++) {
				if (heads[i] == -1) {
					return true;
				}
			}

			return false;
		}
	public:

		vector<string> predicted_labels;
		vector<int> predicted_labels_id;
		vector<int> predicted_heads;
		sparsevec oracle_fv;
		double oracle_prob;
		sparsevec predicted_fv;
		double predicted_prob;

		vector<int> heads;
		vector<int> partial_heads;
		vector<string> deprels;
		vector<int> labels_id;
		


		vector<string> forms;
		vector<int> forms_id;
		vector<int> pret_forms_id;
		vector<string> lemmas;
		vector<string> cpostags;
		vector<int> pos_id;
		
		vector<int> verb_cnt;
		vector<int> conj_cnt;
		vector<int> punc_cnt;

		vector<string> orig_lemmas;
		vector<string> orig_cpostags;
		vector<string> postags;
		vector<string> orig_feats;
		vector<string> pheads;
		vector<string> pdeprels;

		string info;

		bool violate(const int m, const int h) {
				assert(m > 0 && h >= 0);
				if (partial_heads[m] > 0) return true;
				for (int m2 = 1; m2 < partial_heads.size(); ++m2) {
						const int h2 = partial_heads[m2];
						if (h2 < 0) continue;
						if (m2 == h && h2 == m) return true; // reverse
						if (h2 == 0 && h == 0) return true; // multi-root
						const int min1 = min(m,h);
						const int max1 = max(m,h);
						const int min2 = min(m2, h2);
						const int max2 = max(m2, h2);
						if (min1 < min2 && min2 < max1 && max1 < max2) return true; // crossed
						if (min2 < min1 && min1 < max2 && max2 < max1) return true; // crossed
						if (m2 < h && h < m && m == h2) return true;
						if (m < h2 && h2 < m2 && m2 == h) return true;
						if (h2 == m && m < h && h < m2) return true;
						if (h == m2 && m2 < h2 && h2 < m) return true;
				}
				return false;
		}

		// add more partial anno from sys_heads, not violate existing partial_heads
		void add_more_partial_heads(const vector<int> &sys_heads) {
				const int len = size();
				assert(len == partial_heads.size() && len == sys_heads.size());
				for (int i = 1; i < len; ++i) {
						if (partial_heads[i] >= 0) continue;
						if (sys_heads[i] < 0) continue;
						if (!violate(i, sys_heads[i])) partial_heads[i] = sys_heads[i];
				}
		}

		void set_partial_heads(const int i, const string &heads) {
				vector<string> vec;
				simpleTokenize(heads, vec, "_");
				if (vec.empty() || vec[0][0] < '0' || vec[0][0] > '9') {
					//inst->set_candidate_heads_all_possible_heads_for_one_word(inst->candidate_heads_answer, i);
					//inst->copy_candidate_heads_for_one_word(inst->candidate_heads_answer, inst->candidate_heads_base, i);
					//cerr << "IOPipe::preprocessInstance (FEATS): empty filtered heads!" << endl;
					//exit(-1);
				} else {
						if (vec.size() > 1) {
								cerr << "FEATS COLUMN has two many numbers, only partial anno is supported, ambiguous labeling is not" << endl;
								exit(-1);
						}
							const int head_id = toInteger(vec[0]);
							partial_heads[i] = head_id;
							// inst->set_candidate_heads_add_a_head_for_one_word(inst->candidate_heads_answer, i, head_id);
							// add the viterbi 1-best; avoid the candidate heads can not form a tree.
							// inst->candidate_heads_anwser[i][inst->heads[i]] = true;
				}
		}

		void output_instance_to_stderr() const {
			cerr << "***** info about this instance: " << endl
				<< "\tid: " << id << endl
				<< "\tsome thing you should know: " << info << endl;
			for (int i = 1; i < size(); ++i) {
				cerr << i << "\t"
					<< forms[i] << "\t"
					<< orig_lemmas[i] << "\t"
					<< orig_cpostags[i] << "\t"
					<< postags[i] << "\t"
					<< orig_feats[i] << "\t"		
					<< (heads2.empty() ? heads[i] : heads2[i]) << "\t"
					<< deprels[i]	<< "\t" 
					<< pheads[i] << "\t"
					<< pdeprels[i] << endl;
			}
			cerr << endl;
		}
	};

} // namespace dparser


#endif



