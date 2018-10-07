#ifndef __BASELOSS__
#define __BASELOSS__

#include "common.h"
#include "Instance.h"

namespace dparser {

	class BaseLoss {
	public:
		virtual inline dtype loss(BiaffineNode *x, dparser::Instance* inst, int word_num) = 0;
		virtual inline dtype predict_arc_and_rel(BiaffineNode* arc, BiaffineNode* rel, dparser::Instance* inst) = 0;
		virtual inline dtype rel_loss(BiaffineNode* x, const dparser::Instance* inst, int word_num) = 0;

		virtual inline dtype loss(vector<vector<SBiaffineNode>> &x, dparser::Instance* inst, int word_num) = 0;
		virtual inline dtype rel_loss(vector<vector<vector<SBiaffineNode>>> &x, dparser::Instance* inst, int word_num) = 0;
		virtual inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, BiaffineNode* rel, dparser::Instance* inst) = 0;
		virtual inline dtype predict_arc_and_rel(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) = 0;
		virtual inline void constrained_predict(const vector<vector<SBiaffineNode>> &arc, vector<vector<vector<SBiaffineNode>>> &rel, dparser::Instance* inst) = 0;


		virtual inline dtype rel_loss(BiaffineNode* x, const dparser::Instance* inst, int word_num, int which) = 0;
		virtual inline dtype loss(BiaffineNode *x, dparser::Instance* inst, int word_num, int which) = 0;
	};
}




#endif