#include "IOPipe.h"
#include <iterator>

#include "CharUtils.h"

using namespace std;
using namespace egstra;


namespace dparser {

	void IOPipe::preprocessInstance( Instance *inst)
	{
		const int length = inst->size();

		if (_use_lemma) {
			inst->lemmas.resize(length);
			inst->lemmas[0] = NO_FORM;
		}
		if (_filtered_arc) {
			inst->partial_heads.clear();
			inst->partial_heads.resize(length, -1);
		}

		for (int i = 1; i < length; ++i) {
			if (!_labeled) inst->deprels[i] = NO_FORM;

			if (_copy_cpostag_from_postag) {
				inst->cpostags[i] = inst->postags[i];
			} else if (_get_cpostag_from_pdeprel) {
				vector<string> vec;
				simpleTokenize(inst->pdeprels[i], vec, "_");
				if (vec.empty()) {
					cerr << "candidate pos list [pdeprel] empty(): node_i = " << i << endl;
					exit(-1);
				}
				inst->cpostags[i] = vec[0];
			}
			
			if (_use_lemma) {
				if (_english) {
					string form_lc = toLower(inst->forms[i]);
					inst->lemmas[i] = ( form_lc.length()<=5 ? form_lc : form_lc.substr(0,5) );
				} else {
					//assert(!inst->chars[i].empty());
					//inst->lemmas[i] = inst->chars[i].back();
				}
			}
			if (_filtered_arc) inst->set_partial_heads(i, inst->orig_feats[i]);
		}
		
		// set postags for in between features
		inst->verb_cnt.resize(length);
		inst->conj_cnt.resize(length);
		inst->punc_cnt.resize(length);
		inst->verb_cnt[0] = 0;
		inst->conj_cnt[0] = 0;
		inst->punc_cnt[0] = 0;

		for (int i = 1; i < length; ++i) {
			const string &tag = inst->cpostags[i];
			inst->verb_cnt[i] = inst->verb_cnt[i-1];
			inst->conj_cnt[i] = inst->conj_cnt[i-1];
			inst->punc_cnt[i] = inst->punc_cnt[i-1];
			if(tag[0] == 'v' || tag[0] == 'V') {
				++inst->verb_cnt[i];
			} else if(tag == "wp" || tag == "WP" || tag == "Punc" || tag == "PU" || tag == "," || tag == ":") {
				++inst->punc_cnt[i];
			} else if( tag == "Conj" ||	tag == "CC" || tag == "cc" || tag == "c") {
				++inst->conj_cnt[i];
			}
		}
	}

	void IOPipe::getInstancesFromInputFile( const int startId /*= 0*/, const int maxInstNum/*=-1*/, const int instMaxLen/*=-1*/, const int instMinLen/*=1*/ )
	{
		cerr << "Get all instances from " << m_inf_name; print_time();
		dealloc_instance();

		_start_id = startId;

		int inst_thrown_ctr = 0;
		while (1) {
			const size_t this_posi = _inf_current_posi;
			const int this_id = startId + getInstanceNum();

			Instance * const inst = m_reader->getNext(this_id, _inf_current_posi);
			if (!inst) break;
			if (inst->forms.size() != inst->orig_cpostags.size()) {
				cerr << "[BF " << inst_thrown_ctr++ << ":" << inst->size() << "] "; // Wenliang's data
				delete inst;
				continue;
			}

			if (!(check.check_validness(inst) && check.check_validness2(inst))) {
				cerr << "[B " << inst_thrown_ctr++ << ":" << inst->size() << "] "; // Wenliang's data
				delete inst;
				continue;
			}


			if (instMaxLen > 0 && inst->size() > instMaxLen 
                || inst->size()-1 < instMinLen) { // to be consistent with the old version.
				cerr << " [" << inst_thrown_ctr++ << ":" << inst->size() << "] ";
				delete inst;
			} else {
				if (_use_instances_posi) {
					delete inst;
					m_instances_posi.push_back(this_posi);
				} else {
					m_instances.push_back(inst);
					preprocessInstance(inst);
				}
			}

			if (maxInstNum > 0 && getInstanceNum() == maxInstNum) break;
		}
		
		fillVecInstIdxToRead();

		cerr << "\ninstance num: " << getInstanceNum() << endl;
		cerr << "Done!"; print_time();
	}
}


