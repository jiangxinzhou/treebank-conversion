/**
* Contains some string manipulation functions
* @file CharUtils.cc
* @author Mihai Surdeanu
*/

#include <list>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <string>


#include "CharUtils.h"
#include "CppAssert.h"

using namespace std;

namespace egstra {
	bool startsWith(const std::string & big,
		const std::string & small)
	{
		if(small.size() > big.size()) return false;

		for(size_t i = 0; i < small.size(); i ++){
			if(small[i] != big[i]){
				return false;
			}
		}

		return true;
	}

	bool endsWith(const std::string & big,
		const std::string & small)
	{
		size_t bigSize = big.size();
		size_t smallSize = small.size();

		if(smallSize > bigSize) return false;

		for(size_t i = 0; i < smallSize; i ++){
			if(small[i] != big[bigSize - smallSize + i]){
				return false;
			}
		}

		return true;
	}

	int toInteger(const std::string & s)
	{
		return strtol(s.c_str(), NULL, 10);
	}
	
	double toDouble(const std::string & s)
	{
		return strtod(s.c_str(), NULL);
	}

	void toString(const int i, std::string& s) {
		std::ostringstream os;
		os << i;
		s = os.str();
	}

	/*void split_bystr(const std::string &str, std::vector<std::string> &vec, const std::string &sep)
	{
		vec.clear();
		std::string::size_type pos1 = 0, pos2 = 0;
		std::string word;
		while((pos2 = str.find(sep, pos1)) != string::npos)
		{
			word = str.substr(pos1, pos2-pos1);
			pos1 = pos2 + sep.size();
			if(!word.empty()) vec.push_back(word);
		}
		word = str.substr(pos1);
		if(!word.empty()) vec.push_back(word);
	}*/

	void join_bystr(const vector<string> &vec, string &str, const string &sep)
	{
		str = "";
		if (vec.empty()) return;
		str = vec[0];		
		for(int i = 1; i < vec.size(); ++i) str += sep + vec[i];
	}

	void simpleTokenize(const std::string & input,
		std::vector<std::string> & output,
		const std::string & separators)
	{
		output.clear();
		for(int start = input.find_first_not_of(separators);
			start < (int) input.size() && start >= 0;
			start = input.find_first_not_of(separators, start)){
				int end = input.find_first_of(separators, start);
				if(end < 0) end = input.size();
				output.push_back(input.substr(start, end - start));
				start = end;
		}
	}

	std::string toUpper(const std::string & s)
	{
		std::string o = s;

		for(int i = 0; i < (int) o.size(); i ++){
			o[i] = toupper(o[i]);
		}

		return o;
	}

	std::string toLower(const std::string & s)
	{
		std::string o = s;

		for(int i = 0; i < (int) o.size(); i ++){
			o[i] = tolower(o[i]);
		}

		return o;
	}

	std::string stripString(const std::string & s,
		int left,
		int right)
	{
		return s.substr(left, s.size() - left - right);
	}

	//bool emptyLine(const std::string & input)
	//{
	//	for(size_t i = 0; i < input.size(); i ++)
	//		if(! isblank(input[i])) return false;
	//	return true;
	//}

	/*bool my_getline(std::ifstream &inf, std::string &line)
	{
		if (!std::getline(inf, line)) return false;
		int end = line.size() - 1;
		while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
			line.erase(end--);
		}

		return true;
	}*/

	bool contain_uppercase_character(const std::string & s) {
		for(int i = 0; i < (int) s.size(); i ++){
			if (s[i] >= 'A' && s[i] <= 'Z') return true;
		}
		return false;
	}

	bool contain_hyphen(const std::string & s) {
		for(int i = 0; i < (int) s.size(); i ++){
			if (s[i] == '-') return true;
		}
		return false;
	}

	bool contain_number(const std::string & s) {
		for(int i = 0; i < (int) s.size(); i ++){
			if (s[i] >= '0' && s[i] <= '9') return true;
		}
		return false;
	}

	// Borrowed from "Utf.h" of Zpar by Yue Zhang
	int getCharactersFromUTF8String(const std::string &s, std::vector<std::string> &chars) {
		chars.clear();
		unsigned long int idx = 0;
		while (idx < s.length()) {
			if ((s[idx] & 0x80) == 0) {
				chars.push_back(s.substr(idx, 1));
				++idx;
			} else if ((s[idx] & 0xE0) == 0xC0) {
				chars.push_back(s.substr(idx, 2));
				idx += 2;
			} else if ((s[idx] & 0xF0) == 0xE0) {
				chars.push_back(s.substr(idx, 3));
				idx += 3;
			} else {
				std::cerr << "string '" << s << "' not encoded in unicode utf-8" << endl; 
				exit(-1);
			}
		}

		if (idx != s.length()) {
			cerr << "string '" << s << "' not encoded in unicode utf-8" << endl; 
			exit(-1);
		}
		return chars.size();
	}

	/*
	int main(int argc, char ** argv)
	{
	String p = argv[1];
	vector<String> exp;
	expandNumber(p, exp);
	for(int i = 0; (int) i < exp.size(); i ++){
	cout << "\"" << exp[i] << "\"" << " ";
	}
	cout << endl;
	}
	*/
}
