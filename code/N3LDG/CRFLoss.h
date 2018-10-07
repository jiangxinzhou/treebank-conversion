#ifndef _DPARSER_CHART_ITEM_
#define _DPARSER_CHART_ITEM_

#pragma once

#include <stdlib.h>
#include "common.h"
#include "FVec.h"
#include <list>
#include "Instance.h"
#include "BaseLoss.h"
using namespace std;
using namespace egstra;

namespace dparser {

	class ChartItem
	{
	public:
		const int _s, _t;
		const int _comp;
		const int _offset;
		const ChartItem * const _left;
		const ChartItem * const _right;

		const double _prob;
		const list<const fvec *> _fvs; // unnecessary to store this

	public:
		ChartItem(const int comp, const int s, const int t, const int offset,
			const double prob = 0.0, const list<const fvec *> &fvs = list<const fvec *>(),
			const ChartItem * const left = 0, const ChartItem * const right = 0) :
			_comp(comp), _s(s), _t(t), _offset(offset),
			_prob(prob), _fvs(fvs),
			_left(left), _right(right)
		{}

		//ChartItem(const int s) : // for spans like C(s,s)top
		//	_comp(CMP),
		//	_s(s), _t(s), _offset(0),
		//	_prob(0.0), _fvs(),
		//	_left(0), _right(0)
		//{}

		~ChartItem(void) {}

	public:
		// forbid
		ChartItem(const ChartItem &rhs) :
			_comp(rhs._comp),
			_s(rhs._s), _t(rhs._t), _offset(rhs._offset),
			_prob(rhs._prob), _fvs(rhs._fvs),
			_left(rhs._left), _right(rhs._right)
		{
			cerr << "not allow ChartItem::ChartItem(const ChartItem &rhs)" << endl;
			exit(-1);
		}


		ChartItem(const int s) : // for spans like C(s,s)
			_comp(CMP),
			_s(s), _t(s), _offset(0),
			_prob(0.0), _fvs(),
			_left(0), _right(0)
		{}

		ChartItem &operator =(const ChartItem &rhs) {
			cerr << "not allow ChartItem::operator =(const ChartItem &rhs)" << endl;
			exit(-1);
			return *this;
		}
	};


	class CRFLoss : public BaseLoss {
	public:
		NRMat<double> _i_chart_cmp;
		NRMat3d<double> _i_chart_incmp; // NRMat3d -> NRMat (for unlabeled)
		NRMat<double> _o_chart_cmp;
		NRMat3d<double> _o_chart_incmp; // NRMat3d -> NRMat (for unlabeled)

		NRMat3d< const ChartItem * > _chart_incmp;	// N * N
		NRMat< const ChartItem * > _chart_cmp;		// N * N
		NRMat<int> _chart_incmp_s_t_best_label; // s -> t: argmax_l(l \in L}{ score(incmp[s->t, l]) } )


		CRFLoss() {
		}

		~CRFLoss() {
			dealloc();
		}

		inline void dealloc_m2(NRMat< const ChartItem * > &_chart) {
			const ChartItem  **m2 = _chart.c_buf();
			for (int i = 0; i < _chart.size(); ++i, ++m2) {
				const ChartItem * pitem = (*m2);
				if (pitem) delete pitem;
			}
			_chart.dealloc();
		}
		inline void dealloc_m3(NRMat3d< const ChartItem * > &_chart) {
			const ChartItem  **m3 = _chart.c_buf();
			for (int i = 0; i < _chart.size(); ++i, ++m3) {
				const ChartItem * pitem = (*m3);
				if (pitem) delete pitem;
			}
			_chart.dealloc();
		}

		void dealloc() {
			dealloc_m2(_chart_cmp);
			dealloc_m3(_chart_incmp);
			_chart_incmp_s_t_best_label.dealloc();
			_i_chart_cmp.dealloc();
			_i_chart_incmp.dealloc();
			_o_chart_cmp.dealloc();
			_o_chart_incmp.dealloc();
		}

		bool check_validness(Instance * const inst) {
			if (inst->heads.size() == 0) return true;

			inst->set_candidate_heads_max(inst->candidate_heads_base);

			inst->set_candidate_heads_max(inst->candidate_heads_answer);
			for (int m = 1; m <= inst->size() - 1; m++) {
				int h = inst->heads[m];
				if (h == -1) {
					continue;
				}
				else {
					inst->set_candidate_heads_single_head_for_one_word(inst->candidate_heads_answer, m, h);
				}
			}


			inst->p_cand_heads = &inst->candidate_heads_answer;
			if (inst->candidate_heads_answer.empty()) inst->p_cand_heads = &inst->candidate_heads_base;
			const int len = inst->size();
			inst->prob_dep.resize(len, len, 1);
			inst->prob_dep = 1.;
			inside(inst, true);
			inst->prob_dep.dealloc();
			bool valid = !equal_to_negative_infinite(log_Z(inst));

			return valid;
		}


		bool check_validness2(Instance * const inst) {
			if (inst->heads2.size() == 0) return true;

			inst->set_candidate_heads_max(inst->candidate_heads_base);

			inst->set_candidate_heads_max(inst->candidate_heads_answer);
			for (int m = 1; m <= inst->size() - 1; m++) {
				int h = inst->heads2[m];
				if (h == -1) {
					continue;
				}
				else {
					inst->set_candidate_heads_single_head_for_one_word(inst->candidate_heads_answer, m, h);
				}
			}


			inst->p_cand_heads = &inst->candidate_heads_answer;
			if (inst->candidate_heads_answer.empty()) inst->p_cand_heads = &inst->candidate_heads_base;
			const int len = inst->size();
			inst->prob_dep.resize(len, len, 1);
			inst->prob_dep = 1.;
			inside(inst, true);
			inst->prob_dep.dealloc();
			bool valid = !equal_to_negative_infinite(log_Z(inst));

			return valid;
		}

		void inside(const Instance * const inst, const bool constrained)
		{
			const int length = inst->size();
			//cerr << equal_to_negative_infinite(DOUBLE_NEGATIVE_INFINITY) << endl;
			_i_chart_cmp.resize(length, length);
			_i_chart_incmp.resize(length, length, 1);
			_i_chart_cmp = DOUBLE_NEGATIVE_INFINITY;
			_i_chart_incmp = DOUBLE_NEGATIVE_INFINITY;

			const NRMat<bool> &is_head = *(inst->p_cand_heads);
			assert(!constrained || !is_head.empty());

			for (int s = 0; s < length; ++s) {
				_i_chart_cmp[s][s] = LOG_EXP_ZERO;
			}

			for (int width = 1; width < length; ++width) {
				for (int s = 0; s + width < length; ++s) {
					const int t = s + width;

					double log_sum = DOUBLE_NEGATIVE_INFINITY;
					for (int r = s; r < t; ++r) {
						const double a = _i_chart_cmp[s][r];
						const double b = _i_chart_cmp[t][r + 1];
						log_add_if_not_negative_infinite(log_sum, a, b);
					}

					if (!equal_to_negative_infinite(log_sum)) {
						// I(s->t)
						if (!constrained || is_head[t][s]) {
							for (int l = 0; l<1; l++) {
								const double c = inst->prob_dep[s][t][l];
								if (!equal_to_negative_infinite(c)) {
									_i_chart_incmp[s][t][l] = log_sum + c;
								}
							}
						}
						// I(s<-t)
						if (s != 0 && (!constrained || is_head[s][t])) {
							for (int l = 0; l<1; l++) {
								const double c = inst->prob_dep[t][s][l];
								if (!equal_to_negative_infinite(c)) {
									_i_chart_incmp[t][s][l] = log_sum + c;
								}
							}
						}
					}

					// C(s->t)
					if (s != 0 || t == length - 1) { // only consider single-root (vs. multi-root)
						log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = s + 1; r <= t; ++r) {
							double a_sum = DOUBLE_NEGATIVE_INFINITY;
							for (int l = 0; l<1; l++) {
								const double a = _i_chart_incmp[s][r][l];
								//log_add_another(a_sum,a);
								log_add_if_not_negative_infinite(a_sum, a);
							}
							const double b = _i_chart_cmp[r][t];
							log_add_if_not_negative_infinite(log_sum, a_sum, b);
						}
						_i_chart_cmp[s][t] = log_sum;
					}

					if (s != 0) {
						log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = s; r < t; ++r) {
							const double a = _i_chart_cmp[r][s];
							double b_sum = DOUBLE_NEGATIVE_INFINITY;
							for (int l = 0; l<1; l++) {
								const double b = _i_chart_incmp[t][r][l];
								//log_add_another(b_sum,b);
								log_add_if_not_negative_infinite(b_sum, b);
							}
							log_add_if_not_negative_infinite(log_sum, a, b_sum);
						}
						_i_chart_cmp[t][s] = log_sum;
					}
				}
			}
			//cerr << "\nlog_Z: " << log_Z(inst) << endl;
		}

		void get_result(Instance *inst) const
		{
			const int length = inst->size();
			inst->predicted_heads.clear(); // vector<int>
			inst->predicted_heads.resize(length, -1);
			inst->predicted_prob = 0; // actually is score, not prob

			const ChartItem * best_item = _chart_cmp[0][length - 1];
			inst->predicted_prob = best_item->_prob;
			get_best_parse_recursively(inst, best_item);
		}

		void get_best_parse_recursively(Instance *inst, const ChartItem * const item) const
		{
			if (!item) return;
			get_best_parse_recursively(inst, item->_left);

			if (INCMP == item->_comp) {
				// set heads and deprels
				assert(0 > inst->predicted_heads[item->_t]);
				inst->predicted_heads[item->_t] = item->_s;
				//inst->predicted_labels[item->_t] = item->_offset;
			}
			else if (CMP == item->_comp) { // do nothing
			}
			else if (SIB_SP == item->_comp) {
			}
			else {
				cerr << "unknown item->_comp: " << item->_comp << endl;
				exit(0);
			}

			get_best_parse_recursively(inst, item->_right);
		}

		inline  void log_add_if_not_negative_infinite(double &self, const double a) {
			if (!equal_to_negative_infinite(a)) {
				if (equal_to_negative_infinite(self)) {
					self = a;
				}
				else {
					log_add_another(self, a);
				}
			}
		}

		void log_add_if_not_negative_infinite(double &self, const double a, const double b) {
			if (!equal_to_negative_infinite(a) && !equal_to_negative_infinite(b)) {
				if (equal_to_negative_infinite(self)) {
					self = a + b;
				}
				else {
					log_add_another(self, a + b);
				}
			}
		}

		void log_add_if_not_negative_infinite(double &self, const double a, const double b, const double c) {
			if (!equal_to_negative_infinite(a) && !equal_to_negative_infinite(b) && !equal_to_negative_infinite(c)) {
				if (equal_to_negative_infinite(self)) {
					self = a + b + c;
				}
				else {
					log_add_another(self, a + b + c);
				}
			}
		}

		/*void log_add_if_not_negative_infinite(double &self, const double a, const double b, const double c) {
			if (!equal_to_negative_infinite(a) && !equal_to_negative_infinite(b) && !equal_to_negative_infinite(c)) {
				if (equal_to_negative_infinite(self)) {
					self = a + b + c;
				}
				else {
					log_add_another(self, a + b + c);
				}
			}
		}*/

		void log_add_another(double &self, const double another) {
			self = log_add(self, another);
		}

		double log_add(const double a, const double b)
		{
			if (a > b)
				return a + log(1 + exp(b - a));
			else
				return b + log(1 + exp(a - b));
		}

		bool add_item(const ChartItem * &add_place, const ChartItem * const new_item) {
			if (add_place == NULL) {
				add_place = new_item;
				return true;
			}
			if (add_place->_prob < new_item->_prob - EPS) { // absolutely less than the new item
				delete add_place;
				add_place = new_item;
				return true;
			}
			else {
				delete new_item;
				return false;
			}
		}

		void unlabeled_viterbi_decode_projective(const Instance *inst, const NRMat<double> &dep_score_or_prob) {
			const int _label_dim = 1;
			const NRMat<double> &prob_dep = dep_score_or_prob;
			const int length = inst->size();
			_chart_cmp.resize(length, length);
			_chart_incmp.resize(length, length, _label_dim);
			_chart_incmp_s_t_best_label.resize(length, length);
			_chart_incmp_s_t_best_label = -1;
			_chart_cmp = NULL;
			_chart_incmp = NULL;
			for (int i = 0; i < length; i++) {
				_chart_cmp[i][i] = new ChartItem(i);
			}

			const NRMat<bool> &is_head = *(inst->p_cand_heads);

			for (int width = 1; width < length; width++) {
				for (int s = 0; s + width < length; s++) {
					const int t = s + width;
					for (int r = s; r < t; r++) { // C(s,r) + C(t,r+1)
						const ChartItem * const left = _chart_cmp[s][r];
						const ChartItem * const right = _chart_cmp[t][r + 1];
						if (!left || !right) continue;
						if (is_head[t][s]) { // I(s,t)
							for (int l = 0; l < _label_dim; l++) {
								list<const fvec *> fvs;
								double prob = left->_prob + right->_prob;
								prob += prob_dep[s][t];
								const ChartItem * const item = new ChartItem(INCMP, s, t, l, prob, fvs, left, right);
								add_item(_chart_incmp[s][t][l], item); // add_item() will always store the highest-scoring item (as known as the max step); In the inside algorithm, this would be a sum step.
							}
						}
						if (s != 0 && is_head[s][t]) { // I(t,s)
							for (int l = 0; l < _label_dim; l++) {
								list<const fvec *> fvs;
								double prob = left->_prob + right->_prob;
								prob += prob_dep[t][s];
								const ChartItem * const item = new ChartItem(INCMP, t, s, l, prob, fvs, left, right);
								add_item(_chart_incmp[t][s][l], item);
							}
						}
					}

					for (int l = 0; l < _label_dim; l++) {
						if (is_head[t][s]) {
							const ChartItem * const item = _chart_incmp[s][t][l];
							//assert(item);
							if (item) {
								const int best = _chart_incmp_s_t_best_label[s][t];
								if (best < 0 || item->_prob > _chart_incmp[s][t][best]->_prob + EPS) { // the max step
									_chart_incmp_s_t_best_label[s][t] = l;
								}
							}
						}
						if (s != 0 && is_head[s][t]) {
							const ChartItem * const item = _chart_incmp[t][s][l];
							//assert(item);
							if (item) {
								const int best = _chart_incmp_s_t_best_label[t][s];
								if (best < 0 || item->_prob > _chart_incmp[t][s][best]->_prob + EPS) { // the max step
									_chart_incmp_s_t_best_label[t][s] = l;
								}
							}
						}
					}

					for (int r = s; r <= t; r++) {
						if (r != s) { // C(s,t) = I(s,r) + C(r,t)
							if (s == 0 && t != length - 1) continue; // multi-root NOT allowed
							const ChartItem * const right = _chart_cmp[r][t];

							const int best = _chart_incmp_s_t_best_label[s][r];
							const ChartItem * const left = (best < 0 ? 0 : _chart_incmp[s][r][best]);
							if (left && right) {
								list<const fvec *> fvs;
								const double prob = left->_prob + right->_prob;
								const ChartItem * const item = new ChartItem(CMP, s, t, -1, prob, fvs, left, right);
								add_item(_chart_cmp[s][t], item);
							}
						}

						if (r != t && s != 0) { // C(t,s) = C(r,s) + I(t,r)
												//if (_chart_cmp[r][s].nrows() == 0) continue;

							const ChartItem * const left = _chart_cmp[r][s];
							const int best = _chart_incmp_s_t_best_label[t][r];
							const ChartItem * const right = (best < 0 ? 0 : _chart_incmp[t][r][best]);
							if (left && right) {
								list<const fvec *> fvs;
								double prob = left->_prob + right->_prob;

								const ChartItem * const item = new ChartItem(CMP, t, s, -1, prob, fvs, left, right);
								add_item(_chart_cmp[t][s], item);
							}
						}
					}
				}
			}
		}


		void inside_1o_unlabeled(const dparser::Instance * const inst) {
			const bool constrained = true;
			const int length = inst->size();
			//cerr << equal_to_negative_infinite(DOUBLE_NEGATIVE_INFINITY) << endl;
			_i_chart_cmp.resize(length, length);
			_i_chart_incmp.resize(length, length, 1);
			_i_chart_cmp = DOUBLE_NEGATIVE_INFINITY;
			_i_chart_incmp = DOUBLE_NEGATIVE_INFINITY;

			const nr::NRMat<bool> &is_head = *(inst->p_cand_heads);
			assert(!constrained || !is_head.empty());

			for (int s = 0; s < length; ++s) {
				_i_chart_cmp[s][s] = LOG_EXP_ZERO;
			}

			for (int width = 1; width < length; ++width) {
				for (int s = 0; s + width < length; ++s) {
					const int t = s + width;

					double log_sum = DOUBLE_NEGATIVE_INFINITY;
					for (int r = s; r < t; ++r) {
						const double a = _i_chart_cmp[s][r];
						const double b = _i_chart_cmp[t][r + 1];
						log_add_if_not_negative_infinite(log_sum, a, b);
					}

					if (!equal_to_negative_infinite(log_sum)) {
						// I(s->t)
						if (!constrained || is_head[t][s]) {
							{
								const double c = inst->prob_dep[s][t][0];
								if (!equal_to_negative_infinite(c)) {
									_i_chart_incmp[s][t][0] = log_sum + c;
								}
							}
						}
						// I(s<-t)
						if (s != 0 && (!constrained || is_head[s][t])) {
							{
								const double c = inst->prob_dep[t][s][0];
								if (!equal_to_negative_infinite(c)) {
									_i_chart_incmp[t][s][0] = log_sum + c;
								}
							}
						}
					}

					// C(s->t)
					if (s != 0 || t == length - 1) { // only consider single-root (vs. multi-root)
						log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = s + 1; r <= t; ++r) {
							double a_sum = DOUBLE_NEGATIVE_INFINITY;
							{
								const double a = _i_chart_incmp[s][r][0];
								//log_add_another(a_sum,a);
								log_add_if_not_negative_infinite(a_sum, a);
							}
							const double b = _i_chart_cmp[r][t];
							log_add_if_not_negative_infinite(log_sum, a_sum, b);
						}
						_i_chart_cmp[s][t] = log_sum;
					}

					if (s != 0) {
						log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = s; r < t; ++r) {
							const double a = _i_chart_cmp[r][s];
							double b_sum = DOUBLE_NEGATIVE_INFINITY;
							{
								const double b = _i_chart_incmp[t][r][0];
								//log_add_another(b_sum,b);
								log_add_if_not_negative_infinite(b_sum, b);
							}
							log_add_if_not_negative_infinite(log_sum, a, b_sum);
						}
						_i_chart_cmp[t][s] = log_sum;
					}
				}
			}

		}
		void outside_1o_unlabeled(const dparser::Instance * const inst) {
			const bool constrained = true;
			const int length = inst->size();
			const int n = length - 1;
			_o_chart_cmp.resize(length, length);
			_o_chart_incmp.resize(length, length, 1);
			_o_chart_cmp = DOUBLE_NEGATIVE_INFINITY;
			_o_chart_incmp = DOUBLE_NEGATIVE_INFINITY;
			const NRMat<bool> &is_head = *(inst->p_cand_heads);

			_o_chart_cmp[0][n] = LOG_EXP_ZERO;
			{
				_o_chart_incmp[0][n][0] = _i_chart_cmp[n][n] + _o_chart_cmp[0][n]; // 0 = log(1.0)
			}

			for (int width = length - 2; width > 0; --width) {
				for (int s = 0; s + width < length; ++s) {
					const int t = s + width;
					// ----- C(s -> t) -----			
					if (s != 0) { // single root is allowed (when s == 0, t != m_length-1, due to width)
						double log_sum = DOUBLE_NEGATIVE_INFINITY;
						// I(r->s) + C(s->t) = C(r->t)
						for (int r = 0; r < s; ++r) {
							{
								if (r == 0 && t != length - 1) continue; // only C(0 -> m_length-1) is allowed
								const double a = _i_chart_incmp[r][s][0];
								const double b = _o_chart_cmp[r][t];
								log_add_if_not_negative_infinite(log_sum, a, b);
							}
						}

						for (int r = t + 1; r < length; ++r) {	// s != 0
							{
								if (!constrained || is_head[r][s]) { // C(s->t) + C(t+1 <- r) = I(s->r)
									const double a = _i_chart_cmp[r][t + 1];
									const double b = _o_chart_incmp[s][r][0];
									log_add_if_not_negative_infinite(log_sum, a, b, inst->prob_dep[s][r][0]);
								}
								if (!constrained || is_head[s][r]) { // C(s->t) + C(t+1 <- r) = I(s<-r)
									const double a = _i_chart_cmp[r][t + 1];
									const double b = _o_chart_incmp[r][s][0];
									log_add_if_not_negative_infinite(log_sum, a, b, inst->prob_dep[r][s][0]);
								}
							}
						}
						_o_chart_cmp[s][t] = log_sum;
					} // if (s != 0) { // single root is allowed

					  // ----- C(s <- t) -----

					if (s != 0) { // w0 is never a modifier
						double log_sum = DOUBLE_NEGATIVE_INFINITY;
						// C(s <- t) + I(t <- r) = C(s <- r)
						for (int r = t + 1; r < length; ++r) {
							{
								const double a = _i_chart_incmp[r][t][0];
								const double b = _o_chart_cmp[r][s];
								log_add_if_not_negative_infinite(log_sum, a, b);
							}
						}

						for (int r = 0; r < s; ++r) {
							{
								// C(r -> s-1) + C(s <- t) = I(r -> t)
								if (r == 0 && s - 1 != 0) continue; // multi-root not allowed
								if (!constrained || is_head[t][r]) {
									const double a = _i_chart_cmp[r][s - 1];
									const double b = _o_chart_incmp[r][t][0];
									log_add_if_not_negative_infinite(log_sum, a, b, inst->prob_dep[r][t][0]);
								}
								// C(r -> s-1) + C(s <- t) = I(r <- t)
								if (r != 0 &&
									(!constrained || is_head[r][t])) {
									const double a = _i_chart_cmp[r][s - 1];
									const double b = _o_chart_incmp[t][r][0];
									log_add_if_not_negative_infinite(log_sum, a, b, inst->prob_dep[t][r][0]);
								}
							}
						}
						_o_chart_cmp[t][s] = log_sum;
					} // if (s != 0) { // w0 is never a modifier


					  // ----- I(s -> t) -----
					if (!constrained || is_head[t][s]) {
						double log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = t; r < length; ++r) {
							if (s == 0 && r != length - 1) continue; // multi-root not allowed

																	 // I(s -> t) + C(t -> r) = C(s -> r)
							const double a = _i_chart_cmp[t][r];
							const double b = _o_chart_cmp[s][r];
							log_add_if_not_negative_infinite(log_sum, a, b);
						}

						_o_chart_incmp[s][t][0] = log_sum;
					}

					// ----- I(s <- t) -----
					if (s != 0 &&
						(!constrained || is_head[s][t])) {
						double log_sum = DOUBLE_NEGATIVE_INFINITY;
						for (int r = 1; r <= s; ++r) {
							// C(r <- s) + I(s <- t) = C(r <- t)
							const double a = _i_chart_cmp[s][r];
							const double b = _o_chart_cmp[t][r];
							log_add_if_not_negative_infinite(log_sum, a, b);
						}

						_o_chart_incmp[t][s][0] = log_sum;
					}
				}
			}
		}
		/*virtual void decode_projective(const Instance *inst);*/

		void get_and_check_marginal_prob(const dparser::Instance * const inst, nr::NRMat<double> &marg_prob) {
			//cerr << "\nlog_Z: " << log_Z(inst) << endl;
			bool error_occur = false;
			const int len = inst->size();
			marg_prob.resize(len, len);
			marg_prob = 0;
			for (int m = 1; m < len; ++m) {
				double prob = 0.;
				for (int h = 0; h < len; ++h) {
					if (h == m) continue;
					const double tmp = marginal_prob(inst, h, m, 0);
					marg_prob[h][m] = tmp;
					prob += tmp;
					//cerr << tmp << " ";
				}
				//cerr << prob << endl;
				if (!coarse_equal_to(prob, 1.0)) {
					error_occur = true;
					cerr.precision(5);
					cerr << "\\sum_{h}{prob(<h,m>|x)} (m=" << m << ") : " << prob << endl;
				}

			}
			if (error_occur) {
				//inst->output_candidate_heads();
				inst->output_instance_to_stderr();
				cerr << "\nlog_Z: " << log_Z(inst) << endl;
				cerr << "len: " << inst->size() << endl;
				exit(-1);
			}
		}

		double marginal_prob(const dparser::Instance * const inst, int h, int m, int l) const {
			const double a = _i_chart_incmp[h][m][l];
			const double b = _o_chart_incmp[h][m][l];
			if (equal_to_negative_infinite(a) || equal_to_negative_infinite(b)) {
				return 0;
			}
			else {
				const double p = exp(a + b - log_Z(inst));
				if (p > 1.0 + 1e-5) {
					cerr << "prob = " << p << " h=" << h << " m=" << m << " l=" << l << endl;
				}
				return p;
			}
		}

		inline bool coarse_equal_to(const double a, const double b) {
			const double interval = 1e-3;
			return ((a <= b + interval) && (a >= b - interval));
		}

		bool compute_marginals_1o_unlabeled(dparser::Instance * const inst, nr::NRMat<double> &marg_prob) {
			inside_1o_unlabeled(inst);
			if (equal_to_negative_infinite(log_Z(inst))) {
				cerr << "log_Z() == DOUBLE_NEGATIVE_INFINITY!" << endl;
				//cerr << "constrained? " << (constrained ? "yes" : "no") << endl;
				inst->output_instance_to_stderr();
				cerr << "len: " << inst->size() << endl;
				//exit(-1);
				return false;
			}
			outside_1o_unlabeled(inst);
			get_and_check_marginal_prob(inst, marg_prob);
			return true;
		}

		double log_Z(const dparser::Instance * const inst) const {
			return _i_chart_cmp[0][inst->size() - 1];
		}

	public:
		inline void to_implement_compute_all_probs_dep(const vector<vector<SBiaffineNode>> &x, dparser::Instance* inst) {
			int n = inst->size();
			inst->prob_dep.resize(n, n, 1);
			for (int h = 0; h < n; h++) {
				for (int m = 0; m < n; m++) {
					inst->prob_dep[h][m][0] = x[m][h].val.v[0];
				}
			}
		}

	
		inline void to_implement_compute_all_probs_dep(BiaffineNode *x, dparser::Instance* inst) {
			int n = inst->size();
			inst->prob_dep.resize(n, n, 1);
			inst->prob_dep = 0;
			for (int h = 0; h < n; h++) {
				for (int m = 0; m < n; m++) {
					inst->prob_dep[h][m][0] = x->vals[0].mat()(m, h);
				}
			}
		}

		inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, BiaffineNode* rel, dparser::Instance* inst) {
			int label_num = rel->classDim;

			to_implement_compute_all_probs_dep(arc, inst);
			inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
			int len = inst->predicted_heads.size();

			
			const NRMat<double> &head_prob = inst->marg_prob_base;


			for (int i = 0; i < len; i++) {
				double max = head_prob[0][i];
				int index = 0;
				for (int h = 1; h < len; h++) {
					if (head_prob[h][i] > max + 1e-20) {
						index = h;
						max = head_prob[h][i];
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


		inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) {
			int label_num = rel[0][0].size();

			to_implement_compute_all_probs_dep(arc, inst);
			inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
			int len = inst->predicted_heads.size();


			const NRMat<double> &head_prob = inst->marg_prob_base;


			for (int i = 0; i < len; i++) {
				double max = head_prob[0][i];
				int index = 0;
				for (int h = 1; h < len; h++) {
					if (head_prob[h][i] > max + 1e-20) {
						index = h;
						max = head_prob[h][i];
					}
				}
				inst->predicted_heads[i] = index;
			}



			for (int i = 1; i < len; i++) {
				int predict_head = inst->predicted_heads[i];
				int max_index = 0;
			
				dtype max_prob = rel[i][predict_head][0].val.v[0];
				for (int r = 1; r < label_num; r++) {
					if (rel[i][predict_head][r].val.v[0] > max_prob) {
						max_index = r;
						max_prob = rel[i][predict_head][r].val.v[0];
					}
				}

				inst->predicted_labels_id[i] = max_index + 2;
			}

		}

	inline dtype predict_arc_and_rel(BiaffineNode* arc, BiaffineNode* rel, dparser::Instance* inst) {
		to_implement_compute_all_probs_dep(arc, inst);

		inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
		compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
		int len = inst->predicted_heads.size();
		int label_num = rel->classDim;
		const NRMat<double> &head_prob = inst->marg_prob_base;

		for (int i = 1; i < len; i++) {
			double max = head_prob[0][i];
			int index = 0;
			for (int h = 1; h < len; h++) {
				if (head_prob[h][i] > max + 1e-20) {
					index = h;
					max = head_prob[h][i];
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


	inline dtype loss(vector<vector<SBiaffineNode>> &x, dparser::Instance* inst, int word_num) {
		to_implement_compute_all_probs_dep(x, inst);

		inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
		compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
		double loss = log_Z(inst);


		inst->p_cand_heads = &(inst->candidate_heads_answer);
		compute_marginals_1o_unlabeled(inst, inst->marg_prob_answer);
		loss -= log_Z(inst);
		loss /= word_num;


		int n = inst->size();
		for (int h = 0; h < n; h++) {
			for (int m = 1; m < n; m++) {
				if (m == h) continue;
				int head;
				if (inst->type.compare("multi-source") == 0) {
					head = inst->heads2[m];
				}
				else {
					head = inst->heads[m];
				}
				const double this_loss = inst->marg_prob_base[h][m] - inst->marg_prob_answer[h][m];

				if (head != -1) {
					if (h == head) {
						assert(abs(inst->marg_prob_answer[h][m] - (double)1.0) < 1e-10);
					}
					else {
						assert(abs(inst->marg_prob_answer[h][m] - (double)0.0) < 1e-10);
					}
				}
				/*else {
				cerr << h << "\t" << m << "\t" << n << endl;
				cerr << this_loss << "\t" << inst->marg_prob_answer[h][m] << "\t" << inst->marg_prob_base[h][m] << endl;
				cerr << this_loss << "\t" << inst->marg_prob_answer[h][m] << "\t" << inst->marg_prob_base[h][m] << endl;
				if(!(this_loss < 1e-10 || coarse_equal_to(inst->marg_prob_answer[h][m], 0))) {
				inst->output_instance_to_stderr();
				for (int x = 0; x < n; ++x) {
				cerr << x << "\t" << inst->marg_prob_answer[x][m] << "\t" << inst->marg_prob_base[x][m] << endl;
				}
				}
				assert(this_loss < 1e-10 || coarse_equal_to(inst->marg_prob_answer[h][m], 0));
				}
				*/

				x[m][h].loss.v[0] += this_loss / word_num;
			}
		}
	}

		inline dtype loss(BiaffineNode *x, dparser::Instance* inst, int word_num) {
			to_implement_compute_all_probs_dep(x, inst);

			inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
			double loss = log_Z(inst);


			inst->p_cand_heads = &(inst->candidate_heads_answer);
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_answer);
			loss -= log_Z(inst);
			loss /= word_num;

			int n = inst->size();
			for (int h = 0; h < n; h++) {
				for (int m = 1; m < n; m++) {
					if (m == h) continue;
					int head;
					if (inst->type.compare("multi-source") == 0) {
						head = inst->heads2[m];
					}
					else {
						head = inst->heads[m];
					}
					const double this_loss = inst->marg_prob_base[h][m] - inst->marg_prob_answer[h][m];

					if (head != -1) {
						if (h == head) {
							assert(abs(inst->marg_prob_answer[h][m] - (double)1.0) < 1e-10);
						}
						else {
							assert(abs(inst->marg_prob_answer[h][m] - (double)0.0) < 1e-10);
						}
					}
					/*else {
						cerr << h << "\t" << m << "\t" << n << endl;
						cerr << this_loss << "\t" << inst->marg_prob_answer[h][m] << "\t" << inst->marg_prob_base[h][m] << endl;
						cerr << this_loss << "\t" << inst->marg_prob_answer[h][m] << "\t" << inst->marg_prob_base[h][m] << endl;
						if(!(this_loss < 1e-10 || coarse_equal_to(inst->marg_prob_answer[h][m], 0))) {
							inst->output_instance_to_stderr();
							for (int x = 0; x < n; ++x) {
								cerr << x << "\t" << inst->marg_prob_answer[x][m] << "\t" << inst->marg_prob_base[x][m] << endl;
							}
						}
						assert(this_loss < 1e-10 || coarse_equal_to(inst->marg_prob_answer[h][m], 0));
					}
 */

					x->losses[0].mat()(m, h) += this_loss / word_num;
				}
			}
		}


		inline dtype loss(BiaffineNode *x, dparser::Instance* inst, int word_num, int which) {
			to_implement_compute_all_probs_dep(x, inst);


			inst->set_candidate_heads_max(inst->candidate_heads_base);

			inst->set_candidate_heads_max(inst->candidate_heads_answer);
			for (int m = 1; m <= inst->size() - 1; m++) {
				int h = (which == 0) ? inst->heads[m] : inst->heads2[m];
				if (h == -1) {
					continue;
				}
				else {
					inst->set_candidate_heads_single_head_for_one_word(inst->candidate_heads_answer, m, h);
				}
			}


			inst->p_cand_heads = &(inst->candidate_heads_base); // 假设inst->candidate_heads_base已经设置好!!
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_base); // inst->marg_prob_base保存整个搜索空间的概率分布
			double loss = log_Z(inst);


			inst->p_cand_heads = &(inst->candidate_heads_answer);
			compute_marginals_1o_unlabeled(inst, inst->marg_prob_answer);
			loss -= log_Z(inst);
			loss /= word_num;

			int n = inst->size();
			for (int h = 0; h < n; h++) {
				for (int m = 1; m < n; m++) {
					if (m == h) continue;
					int head;
					if (which == 0) {
						head = inst->heads[m];
					}
					else {
						head = inst->heads2[m];
					}
					const double this_loss = inst->marg_prob_base[h][m] - inst->marg_prob_answer[h][m];

					if (head != -1) {
						if (h == head) {
							assert(abs(inst->marg_prob_answer[h][m] - (double)1.0) < 1e-10);
						}
						else {
							assert(abs(inst->marg_prob_answer[h][m] - (double)0.0) < 1e-10);
						}
					}

					x->losses[0].mat()(m, h) += this_loss / word_num;
				}
			}
		}

		inline dtype rel_loss(vector<vector<vector<SBiaffineNode>>> &x, dparser::Instance* inst, int word_num) {
			int len = inst->size();
			vector<int> heads = (inst->type.compare("multi-source") == 0) ? inst->heads2 : inst->heads;
			vector<int> labels_id = (inst->type.compare("multi-source") == 0) ? inst->labels_id2 : inst->labels_id;


			int label_num = x[0][0].size();
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
						rel_prob[i][j][r] = x[i][j][r].val.v[0];
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
					x[i][head][r].loss.v[0] += loss / word_num;
				}
			}
		}
		//!this is not crf-loss, but softmax loss
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

		inline dtype rel_loss(BiaffineNode* x, const dparser::Instance* inst, int word_num, int which) {
			int len = inst->size();
			vector<int> heads = (which == 0) ? inst->heads : inst->heads2;
			vector<int> labels_id = (which == 0) ? inst->labels_id : inst->labels_id2;


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

		inline void constrained_predict(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) {
			to_implement_compute_all_probs_dep(arc, inst);

			//if (!inst->heads.empty()) { // completion
				inst->p_cand_heads = &(inst->candidate_heads_answer);
				compute_marginals_1o_unlabeled(inst, inst->marg_prob_answer);
				unlabeled_viterbi_decode_projective(inst, inst->marg_prob_answer);
				get_result(inst);
			//}
			//else {  // all predicted
				//inst->p_cand_heads = &(inst->candidate_heads_base);
				//compute_marginals_1o_unlabeled(inst, inst->marg_prob_base);
				//unlabeled_viterbi_decode_projective(inst, inst->marg_prob_base);
				//get_result(inst);
			//}

			int len = inst->predicted_heads.size();
			int label_num = rel[0][0].size();
			for (int i = 1; i < len; i++) {
				int predict_head = inst->predicted_heads[i];
				int max_index = 0;

				dtype max_prob = rel[i][predict_head][0].val.v[0];
				for (int r = 1; r < label_num; r++) {
					if (rel[i][predict_head][r].val.v[0] > max_prob) {
						max_index = r;
						max_prob = rel[i][predict_head][r].val.v[0];
					}
				}

				inst->predicted_labels_id[i] = max_index + 2;
			}
		}
	};

	
}


#endif


