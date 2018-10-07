#include "CONLLWriter.h"

#include <vector>
#include <sstream>
using namespace std;

namespace dparser {

	int CONLLWriter::write( const Instance *inst )
	{
		if (!m_outf.is_open()) return -1;

		for (int i = 1; i < inst->size(); ++i) {
			m_outf << i << "\t"
				<< inst->forms[i] << "\t"
				<< inst->orig_lemmas[i] << "\t"
				//<< inst->orig_cpostags[i] << "\t"
				<< inst->postags[i] << "\t"
				//<< inst->orig_feats[i] << "\t"		
				<< inst->heads[i] << "\t" 
				<< inst->deprels[i] << "\t"
				<< (inst->predicted_heads.empty() ? inst->heads[i] : inst->predicted_heads[i]) << "\t"
				<< (inst->predicted_labels.empty()? inst->deprels[i]:inst->predicted_labels[i]) << "\t" << endl;
				//<< inst->deprels[i]	<< "\t" 
				//<< inst->pheads[i] << "\t"
				//<< inst->pdeprels[i] << endl;
		}
		m_outf << endl;

		return 0;
	}

}


