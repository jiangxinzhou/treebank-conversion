#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3LDG.h"

using namespace std;

class Options {
public:

	int wordCutOff;
	int featCutOff;
	int charCutOff;
	dtype initRange;
	int maxIter;
	int batchSize;

	dtype adaEps;
	dtype adaAlpha;
	dtype momentum;
	dtype nnRegular;
	dtype dropProb;

	dtype clip;

	int segHiddenSize;
	int hiddenSize;
	int rnnHiddenSize;

	

	int wordcontext;
	bool wordEmbFineTune;

	
	
	

	int charEmbSize;
	int charcontext;
	bool charEmbFineTune;
	int charhiddenSize;

	int typeEmbSize;
	bool typeEmbFineTune;

	int maxsegLen;

	int verboseIter;
	bool saveIntermediate;
	bool train;
	int maxInstance;
	vector<string> testFiles;
	string outBest;
	bool seg;
	int relu;
	int atomLayers;
	int rnnLayers;

	int beam;
	int maxlength;

	//embedding files
	string wordFile;
	string charFile;
	string segFile;
	vector<string> typeFiles;


	/*-------------------------------  */
	int word_emb_size;
	int tag_emb_size;
	int lstm_output_size;
	int mlp_size;
	int mlp_rel_size;

	int tree_lstm_output_size;
	int label_emb_size;

	int pattern_emb_size;
	int m_label_emb_size;
	int h_label_emb_size;
	int lcn_label_emb_size;
	string range;
	/*-------------------------------  */

	Options() 
	{
		wordCutOff = 0;
		featCutOff = 0;
		charCutOff = 0;
		initRange = 0.01;
		maxIter = 1000;
		batchSize = 100000;
		adaEps = 1e-6;
		adaAlpha = 0.01;
		momentum = 0.9;
		nnRegular = 1e-8;
		dropProb = 0.5;
		
		clip = 15;
		lstm_output_size = 0;
		tree_lstm_output_size = 0;
		mlp_size = 0;
		mlp_rel_size = 0;

		segHiddenSize = 100;
		hiddenSize = 200;
		rnnHiddenSize = 100;
		wordcontext = 2;
		wordEmbFineTune = true;
		charEmbSize = 50;
		charcontext = 2;
		charEmbFineTune = true;
		charhiddenSize = 50;

		typeEmbSize = 50;
		typeEmbFineTune = true;

		verboseIter = 100;
		saveIntermediate = true;
		train = false;
		maxInstance = -1;
		testFiles.clear();
		outBest = "";
		relu = 0;
		seg = false;
		atomLayers = 1;
		rnnLayers = 1;
		maxsegLen = 5;

		beam = 64;
		maxlength = 256;

		wordFile = "";
		charFile = "";
		segFile = "";
		typeFiles.clear();

		word_emb_size = -1;
		tag_emb_size = -1;
		lstm_output_size = -1;
		mlp_size = -1;
		mlp_rel_size = -1; 

		tree_lstm_output_size = -1;
		label_emb_size = -1;

		pattern_emb_size = -1;
		m_label_emb_size = -1;
		h_label_emb_size = -1;
		lcn_label_emb_size = -1;
		range = "";
	}

	virtual ~Options() {}

	void setOptions(const vector<string> &vecOption) {
		int i = 0;
		for (; i < vecOption.size(); ++i) {
			pair<string, string> pr;
			string2pair(vecOption[i], pr, '=');
			if (pr.first == "word_emb_size") word_emb_size = atoi(pr.second.c_str());
			if (pr.first == "tag_emb_size")  tag_emb_size = atoi(pr.second.c_str());
			if (pr.first == "lstm_output_size")  lstm_output_size = atoi(pr.second.c_str());
			if (pr.first == "mlp_size")  mlp_size = atoi(pr.second.c_str());
			if (pr.first == "mlp_rel_size") mlp_rel_size = atoi(pr.second.c_str());

			if (pr.first == "tree_lstm_output_size") tree_lstm_output_size = atoi(pr.second.c_str());
			if (pr.first == "label_emb_size") label_emb_size = atoi(pr.second.c_str());

			if (pr.first == "pattern_emb_size") pattern_emb_size = atoi(pr.second.c_str());
			if (pr.first == "m_label_emb_size") m_label_emb_size = atoi(pr.second.c_str());
			if (pr.first == "h_label_emb_size") h_label_emb_size = atoi(pr.second.c_str());
			if (pr.first == "lcn_label_emb_size") lcn_label_emb_size = atoi(pr.second.c_str());
			if (pr.first == "range") range = pr.second.c_str();
			
			
			if (pr.first == "batchSize") batchSize = atoi(pr.second.c_str());
			if (pr.first == "clip") clip = atof(pr.second.c_str());

			if (pr.first == "adaEps") adaEps = atof(pr.second.c_str());
			if (pr.first == "adaAlpha") adaAlpha = atof(pr.second.c_str());
			if (pr.first == "nnRegular") nnRegular = atof(pr.second.c_str());
		}
	}

	void showOptions() {
		std::cout << "word_emb_size = " << word_emb_size << std::endl;
		std::cout << "tag_emb_size = " << tag_emb_size << std::endl;
		std::cout << "lstm_output_size = " << lstm_output_size << std::endl;
		std::cout << "mlp_size = " << mlp_size << std::endl;
		std::cout << "mlp_rel_size = " << mlp_rel_size << std::endl;

		std::cout << "tree_lstm_output_size = " << tree_lstm_output_size << std::endl;
		std::cout << "label_emb_size = " << label_emb_size << std::endl;

		std::cout << "pattern_emb_size = " << pattern_emb_size << std::endl;
		std::cout << "m_label_emb_size = " << m_label_emb_size << std::endl;
		std::cout << "h_label_emb_size = " << h_label_emb_size << std::endl;
		std::cout << "lcn_label_emb_size = " << lcn_label_emb_size << std::endl;
		std::cout << "range = " << range << std::endl;

		std::cout << "batchSize = " << batchSize << std::endl;
		std::cout << "adaEps = " << adaEps << std::endl;
		std::cout << "adaAlpha = " << adaAlpha << std::endl;
		std::cout << "nnRegular = " << nnRegular << std::endl;
		std::cout << "clip = " << clip << std::endl;
		std::cout << "range = " << range << std::endl;
	}

	void load(const std::string& infile) {
		ifstream inf;
		inf.open(infile.c_str());
		if (!inf.is_open()){
			cerr << "open option file error! " << endl;
		}
		vector<string> vecLine;
		while (1) {
			string strLine;
			if (!my_getline(inf, strLine)) {
				break;
			}
			if (strLine.empty())
				continue;
			vecLine.push_back(strLine);
		}
		inf.close();
		setOptions(vecLine);
	}
};

#endif

