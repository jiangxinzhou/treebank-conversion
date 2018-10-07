#ifndef _FEATURE_EXTRACTER_
#define _FEATURE_EXTRACTER_
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <list>
#include <map>

#include "Instance.h"
#include "common.h"
#include "CharUtils.h"
#include "NRMat.h"

#include "FVec.h"
#include "basic-features/FeatureDictionary.h"

using namespace egstra;
using namespace std;
using namespace nr;

namespace dparser {

	class FGen
	{
	public:
		FeatureDictionary _label_dict;
		FeatureDictionary _word_dict;
		FeatureDictionary _pos_dict;
		NRVec<const char *> _label_id_2_str;
	private:
		string _name;
		bool _generation_mode;

	private: // options

		int _fcutoff; // only use features with freq >= _fcutoff 

	public:
		FGen() {
			_name = "FGen";
			_generation_mode = false;

			_g_label_num = 0;
			_g_feat_dim = 0;
		}

		~FGen() {}
		
		void process_options();
		void start_generation_mode() { _generation_mode = true; }
		void stop_generation_mode() { _generation_mode = false; }

		void save_dictionaries(const string &dictdir, int minCount) /*const*/;
		void load_dictionaries(const string &dictdir);

		int get_label_id(const string &pos) {
			int id = _label_dict.getFeature(pos,  _generation_mode);
			if (id < 0) {
				id =  _label_dict.getFeature(UNKNOWN, _generation_mode);
			}
			return id;		
		}

		int get_word_id(const string &pos) {
			int id = _word_dict.getFeature(pos, _generation_mode);
			if (id < 0) {
				id = _word_dict.getFeature(UNKNOWN, _generation_mode);
			}
			return id;		
		}

		int get_pos_id(const string &pos) {
			int id = _pos_dict.getFeature(pos, _generation_mode);
			if (id < 0) {
				id = _pos_dict.getFeature(UNKNOWN, _generation_mode);
			}
			return id;		
		}

		int label_id_dummy() {
			return get_label_id(NO_FORM);
		}

		const char *label_id_2_str(const int label_id) const {
			assert(label_id >= 0 && label_id < _label_id_2_str.size());
			return _label_id_2_str[label_id];
		}

		void assign_predicted_label_str(Instance * const inst, const vector<int> &labels_id) const {
			const int len = inst->size();
			assert(len == labels_id.size());
			inst->predicted_labels.resize(len);
			for (int i = 1; i < len; ++i) {
				inst->predicted_labels[i] = label_id_2_str(labels_id[i]);
			}
		}

		void get_labels_id(Instance * inst) {
			const int len = inst->size();
			inst->labels_id.resize(len);
			inst->labels_id[0] = -1;
			for (int i = 1; i < len; ++i) {
				inst->labels_id[i] = get_label_id(inst->deprels[i]);
			}
		}

		void get_words_id(Instance * inst) {
			const int len = inst->size();
			inst->forms_id.resize(len);
			inst->forms_id[0] = -1;
			int unknown_id = get_word_id(UNKNOWN);
			for (int i = 0; i < len; ++i) {
				int word_id = get_word_id(inst->forms[i]);
				inst->forms_id[i] = word_id == -1 ? unknown_id : word_id;
			}
		}

		void get_POS_id(Instance * inst) {
			const int len = inst->size();
			inst->pos_id.resize(len);
			inst->pos_id[0] = -1;
			for (int i = 0; i < len; ++i) {
				inst->pos_id[i] = get_pos_id(inst->cpostags[i])==-1?get_pos_id(UNKNOWN):get_pos_id(inst->cpostags[i]);
			}
		}
		void usage(const char * const mesg) const;
	};

} // namespace gparser_space

#endif


