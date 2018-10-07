#include "FGen.h"
#include <iterator>
#include "StringMap.h"
#include "Util-options.h"
#include "CharUtils.h"
#include "CppAssert.h"

using namespace egstra;
using namespace std;

const string PRP = "PRP";
const string PRP2 = "PRP$";

namespace dparser {
	void FGen::process_options()
	{
		int tmp;
		string strtmp;
		_fcutoff = 1;
		if(options::get("fcutoff", tmp)) {
			_fcutoff = tmp;
		}
	}

	void FGen::usage(const char* const mesg) const {
		cerr << _name << " options:" << endl;
	}

	void FGen::save_dictionaries( const string &dictdir ,int minCount) /*const*/
	{
		assert(!_generation_mode);
		cerr << _name << " : saving feature dictionaries to \""
			<< dictdir << "\"" << endl;

		_word_dict.saveAfterSort(dictdir + "/words.gz", minCount);
		_pos_dict.save(dictdir + "/pos.gz");
		_label_dict.save(dictdir + "/labels.gz");
	}


	void FGen::load_dictionaries( const string &dictdir )
	{
		assert(!_generation_mode);
		cerr << _name << " : loading feature dictionaries from \""
			<< dictdir << "\""; print_time();
	
		_word_dict.load(dictdir + "/words.gz", 0);
		_pos_dict.load(dictdir + "/pos.gz", 0);
		_label_dict.load(dictdir + "/labels.gz", 0 );

		_g_label_num = _label_dict.dimensionality();
		_label_id_2_str.resize(_g_label_num );
		_label_id_2_str = NULL;
		_label_dict.collect_keys(_label_id_2_str.c_buf(), _g_label_num);
		
		NULLstr_word_id = get_word_id(NULLstr);
		NULLstr_pos_id = get_pos_id(NULLstr);
		NULLstr_label_id = get_label_id(NULLstr);
	
		cerr << "word number: " << _word_dict.dimensionality() << endl;
		cerr << "pos number: " << _pos_dict.dimensionality() << endl;
		cerr << "label number: " << _g_label_num << endl;
		cerr << "\n done!" << endl; 
		print_time();
	}
} // namespace gparser_space
