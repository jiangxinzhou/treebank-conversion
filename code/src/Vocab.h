#ifndef _VOCAB_
#define _VOCAB_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "MyLib.h"
#include "IOPipe.h"

using namespace dparser;
using namespace std;

class Vocab {
public:
	unordered_map<std::string, int> word_counter;
	unordered_map<std::string, int> word2id;
	unordered_map<std::string, int> tag2id;
	unordered_map<std::string, int> label2id;
	unordered_map<std::string, int> hlt_label2id;


	std::vector<std::string> id2word;
	std::vector<std::string> id2tag;
	std::vector<std::string> id2label;
	std::vector<std::string> id2hlt_label;

	int words_in_train;

	unordered_map<std::string, int> pattern_count;
	unordered_map<std::string, int> pattern2id;
	std::vector<std::string> id2pattern;

	unordered_map<std::string, int> label_pattern_count;
	unordered_map<std::string, int> label_pattern2id;
	std::vector<std::string> id2label_pattern;

public:
	void init(const string& input_file, const string &pret_file, int min_occur_count) {
		ifstream fin(input_file);
		string strLine;
		vector<string> tokens;


		tag2id.insert(std::pair<std::string, int>("<unk>", 0));
		tag2id.insert(std::pair<std::string, int>("<root>", 1));
		label2id.insert(std::pair<std::string, int>("<unk>", 0));
		label2id.insert(std::pair<std::string, int>("<root>", 1));
		hlt_label2id.insert(std::pair<std::string, int>("<unk>", 0));
		hlt_label2id.insert(std::pair<std::string, int>("<root>", 1));
		while (my_getline(fin, strLine)) {
			if (!strLine.empty()) {
				split_bychars(strLine, tokens, "\t");

				//统计词频
				unordered_map<std::string, int>::const_iterator it = word_counter.find(tokens[1]);
				if (it == word_counter.end()) {
					word_counter.insert(std::pair<std::string, int>(tokens[1], 1));
				}
				else {
					word_counter[tokens[1]] += 1;
				}


				it = tag2id.find(tokens[3]);
				if (it == tag2id.end()) {
					tag2id.insert(std::pair<std::string, int>(tokens[3], tag2id.size()));
				}

				it = label2id.find(tokens[7]);
				if (it == label2id.end()) {
					label2id.insert(std::pair<std::string, int>(tokens[7], label2id.size()));
				}

				if (tokens[5].compare("none") != 0 && tokens[5].compare("_") != 0) {
					it = hlt_label2id.find(tokens[5]);
					if (it == hlt_label2id.end()) {
						hlt_label2id.insert(std::pair<std::string, int>(tokens[5], hlt_label2id.size()));
					}
				}
			}
		}

		id2word.push_back("<unk>");
		id2word.push_back("<root>");
		for (unordered_map<std::string, int>::const_iterator it = word_counter.begin(); it != word_counter.end(); it++) {
			if (it->second > min_occur_count) {
				id2word.push_back(it->first);
			}
		}

		add_pret_words(pret_file);

		fin.close();
	}

	void init(const vector<string>& input_files, const string &pret_file, int min_occur_count) {
		tag2id.insert(std::pair<std::string, int>("<unk>", 0));
		tag2id.insert(std::pair<std::string, int>("<root>", 1));
		label2id.insert(std::pair<std::string, int>("<unk>", 0));
		label2id.insert(std::pair<std::string, int>("<root>", 1));
		hlt_label2id.insert(std::pair<std::string, int>("<unk>", 0));
		hlt_label2id.insert(std::pair<std::string, int>("<root>", 1));
		
		id2word.push_back("<unk>");
		id2word.push_back("<root>");

		for(int i=0; i<input_files.size(); i++) {
			ifstream fin(input_files[i]);
			string strLine;
			vector<string> tokens;


			while (my_getline(fin, strLine)) {
				if (!strLine.empty()) {
					split_bychars(strLine, tokens, "\t");

					//统计词频
					unordered_map<std::string, int>::const_iterator it = word_counter.find(tokens[1]);
					if (it == word_counter.end()) {
						word_counter.insert(std::pair<std::string, int>(tokens[1], 1));
					}
					else {
						word_counter[tokens[1]] += 1;
					}


					it = tag2id.find(tokens[3]);
					if (it == tag2id.end()) {
						tag2id.insert(std::pair<std::string, int>(tokens[3], tag2id.size()));
					}

					it = label2id.find(tokens[7]);
					if (it == label2id.end()) {
						label2id.insert(std::pair<std::string, int>(tokens[7], label2id.size()));
					}

					if (tokens[5].compare("none") != 0 && tokens[5].compare("_") != 0) {
						it = hlt_label2id.find(tokens[5]);
						if (it == hlt_label2id.end()) {
							hlt_label2id.insert(std::pair<std::string, int>(tokens[5], hlt_label2id.size()));
						}
					}
				}
			}

			fin.close();
		}

		for (unordered_map<std::string, int>::const_iterator it = word_counter.begin(); it != word_counter.end(); it++) {
			if (it->second > min_occur_count) {
				id2word.push_back(it->first);
			}
		}

		add_pret_words(pret_file);

		
	}

	void add_pret_words(const string &pret_file) {
		words_in_train = id2word.size();

		ifstream fin(pret_file);
		string strLine;
		vector<string> tokens;

		unordered_map<std::string, int> words_in_train_data;
		for (int i = 0; i < id2word.size(); i++) {
			words_in_train_data.insert(std::pair<std::string, int>(id2word[i], 1));
		}


		while (my_getline(fin, strLine)) {
			if (!strLine.empty()) {
				split_bychars(strLine, tokens, " ");

				unordered_map<std::string, int>::const_iterator it = words_in_train_data.find(tokens[0]);
				if (it == words_in_train_data.end()) {
					id2word.push_back(tokens[0]);
				}
			}
		}


		fin.close();
	}

	void save(const string &dict_path) {
		ofstream wfout(dict_path + "/words.gz");
		ofstream tfout(dict_path + "/tags.gz");
		ofstream lfout(dict_path + "/labels.gz");
		ofstream hfout(dict_path + "/hlt-labels.gz");

		wfout << words_in_train << endl;		
		wfout << id2word.size() << endl;
		for (int i = 0; i < id2word.size(); i++) {
			wfout << id2word[i] << " " << i << endl;
		}

		tfout << tag2id.size() << endl;
		for (unordered_map<std::string, int>::const_iterator it = tag2id.begin(); it != tag2id.end(); it++) {
			tfout << it->first << " " << it->second << endl;
		} 

		lfout << label2id.size() << endl;
		for (unordered_map<std::string, int>::const_iterator it = label2id.begin(); it != label2id.end(); it++) {
			lfout << it->first << " " << it->second << endl;
		}

		hfout << hlt_label2id.size() << endl;
		for (unordered_map<std::string, int>::const_iterator it = hlt_label2id.begin(); it != hlt_label2id.end(); it++) {
			hfout << it->first << " " << it->second << endl;
		}

		wfout.close();
		tfout.close();
		lfout.close();
		hfout.close();
	}

	void load(const string &dict_path) {
		ifstream wfin(dict_path + "/words.gz");
		ifstream tfin(dict_path + "/tags.gz");
		ifstream lfin(dict_path + "/labels.gz");
		ifstream hfin(dict_path + "/hlt-labels.gz");
		int word_size, tag_size, label_size, hlt_label_size;

		wfin >> words_in_train;
		wfin >> word_size;
		tfin >> tag_size;
		lfin >> label_size;
		hfin >> hlt_label_size;
		
		cout << "words_in_train: " << words_in_train << endl;
		cout << "word_size: " << word_size << endl;
		GetDict(wfin, word_size, id2word);
		GetDict(tfin, tag_size, id2tag);
		GetDict(lfin, label_size, id2label);
		GetDict(hfin, hlt_label_size, id2hlt_label);

		wfin.close();
		tfin.close();
		lfin.close();
		hfin.close();
		

		reverse(id2word, word2id);
		reverse(id2tag, tag2id);
		reverse(id2label, label2id);
		reverse(id2hlt_label, hlt_label2id);
	}

	void reverse(const vector<string> &v, unordered_map<std::string, int>& m) {
		m.clear();
		for (int i = 0; i < v.size(); i++) {
			m.insert(std::pair<std::string, int>(v[i], i));
		}
	}

	void GetDict(ifstream &fin, int size, vector<string> &v) {
		v.clear();
		v.resize(size);
		string strLine, str;
		int id;

		while (my_getline(fin, strLine)) {
			if (!strLine.empty()) {
				istringstream is(strLine);
				is >> str >> id;
				v[id] = str;
			}
		}
	}

	void get_pret_embs(vector<vector<double>> &embs, const string &pret_file) {
		ifstream fin(pret_file);
		
		string strLine, curWord;
		int wordId;

		vector<string> sLines;
		sLines.clear();
		while (1) {
			if (!my_getline(fin, strLine)) {
				break;
			}
			if (!strLine.empty()) {
				sLines.push_back(strLine);
			}
		}
		fin.close();

		//find the first line, decide the wordDim;
		vector<string> vecInfo;
		split_bychar(sLines[0], vecInfo, ' ');
		int nDim = vecInfo.size() - 1;

		int embs_cnt = id2word.size();
		embs.resize(embs_cnt);
		for (int i = 0; i < embs_cnt; i++) {
			embs[i].resize(nDim);
			for (int j = 0; j < nDim; j++) {
				embs[i][j] = 0.0;
			}
		}

		bool bHasUnknown = false;
		unordered_set<int> indexers;
		vector<double> sum(nDim, 0.0);
		for (int idx = 0; idx < sLines.size(); idx++) {
			split_bychar(sLines[idx], vecInfo, ' ');
			if (vecInfo.size() != nDim + 1) {
				std::cout << "error embedding file" << std::endl;
			}
			curWord = vecInfo[0];
			int wordId = word2id[curWord];
			assert(wordId >= 0);
			for (int idy = 0; idy < nDim; idy++) {
				double curValue = atof(vecInfo[idy + 1].c_str());
				embs[wordId][idy] = curValue;
			}
		}

		int totalnum = 0;
		double sumAll = 0.0;
		for (int idx = 0; idx < embs_cnt; idx++) {
				for (int idy = 0; idy < nDim; idy++) {
					sumAll += embs[idx][idy];
					totalnum++;
			}
		}

		double aver = sumAll / totalnum;
		double devi = 0.0;
		for (int id = 0; id < embs_cnt; id++) {
			for (int idy = 0; idy < nDim; idy++) {
				devi += (embs[id][idy] - aver)*(embs[id][idy] - aver);
			}
		}
		devi = devi / totalnum;
		devi = sqrt(devi);
		for (int i = 0; i < embs_cnt; i++) {
			for (int j = 0; j < nDim; j++) {
				embs[i][j] /= devi;
			}
		}
	}
	
	int get_word_id(const string& word) {
		unordered_map<std::string, int>::const_iterator it = word2id.find(word);
		if (it == word2id.end()) {
			return word2id["<unk>"];
		}
		else {
			return it->second;
		}
	}

	int get_tag_id(const string& tag) {
		unordered_map<std::string, int>::const_iterator it = tag2id.find(tag);
		if (it == tag2id.end()) {
			return tag2id["<unk>"];
		}
		else {
			return it->second;
		}
	}

	int get_label_id(const string& label) {
		unordered_map<std::string, int>::const_iterator it = label2id.find(label);
		if (it == label2id.end()) {
			cout << "unk label!" << endl;
			return label2id["<unk>"];
		}
		else {
			return it->second;
		}
	}

	int get_hlt_label_id(const string& hlt_label) {
		if(hlt_label.compare("none") == 0) return -1;


		unordered_map<std::string, int>::const_iterator it = hlt_label2id.find(hlt_label);
		if (it == hlt_label2id.end()) {
			
			cout << "unk hlt-label! " <<  hlt_label  << endl;
			return label2id["<unk>"];
		}
		else {
			return it->second;
		}
	}
	

	int word_size() const {
		return id2word.size();
	}

	int tag_size() const {
		return id2tag.size();
	}

	int label_size() const {
		return id2label.size();
	}

	int hlt_label_size() const {
		return id2hlt_label.size();
	}
};

#endif
