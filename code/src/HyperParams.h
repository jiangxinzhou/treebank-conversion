#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"
#include "Util-options.h"

using namespace std;

struct HyperParams{
	//required
	

	int batch;
	int maxlength;

	/*-----------emb + lstm + mlp-------*/
	int word_emb_size;
	int tag_emb_size;
	int lstm_output_size;
	int mlp_size;
	int mlp_rel_size;

	/*-----------pattern---------------*/
	int m_label_emb_size;
	int h_label_emb_size;
	int lcn_label_emb_size;
	int pattern_emb_size;
	vector<int> range;

	/*-----------treeLSTM---------------*/
	int label_emb_size;
	int tree_lstm_output_size;
	
	double dropOut;
	double nnRegular; // for optimization
	double adaAlpha;  // for optimization
	double adaEps; // for optimization
	double momentum;
	double clip;
	
	int inputsize;
	int labelSize;
private:
	bool bAssigned;

public:
	HyperParams(){
	}

public:
	void setRequared(Options& opt){
		word_emb_size = opt.word_emb_size;
		tag_emb_size = opt.tag_emb_size;
		lstm_output_size = opt.lstm_output_size;
		mlp_size = opt.mlp_size;
		mlp_rel_size = opt.mlp_rel_size;

		pattern_emb_size = opt.pattern_emb_size;
		m_label_emb_size = opt.m_label_emb_size;
		h_label_emb_size = opt.h_label_emb_size;
		lcn_label_emb_size = opt.lcn_label_emb_size;

		label_emb_size = opt.label_emb_size;
		tree_lstm_output_size = opt.tree_lstm_output_size;

		/* set dist range  */
		string str_range = opt.range;
		vector<string> tokens;
		
		cout << "range: " << str_range << endl;
		split_bychars(str_range, tokens, "$");
		range.clear();
		for (int i = 0; i < tokens.size(); i++) {
			range.push_back(atoi(tokens[i].c_str()));
		}
		for (int i = 0; i < tokens.size(); i++) {
			cout << range[i] << endl;
		}
		/* set dist range  */

		clip = opt.clip;
		batch = opt.batchSize;

		/* update params */
		nnRegular = opt.nnRegular;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		/* update params */

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}

public:

	void print(){

	}
};
#endif SRC_HyperParams_H_
