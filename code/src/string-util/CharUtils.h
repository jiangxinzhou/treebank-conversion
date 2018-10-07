/**
* Contains some string manipulation functions
* @file CharUtils.h
* @author Mihai Surdeanu
*/

#include <string>
#include <vector>
#include <sstream>

#ifndef EGSTRA_CHAR_UTILS_H
#define EGSTRA_CHAR_UTILS_H

namespace egstra {

	//void split_bystr(const std::string &str, std::vector<std::string> &vec, const std::string &sep);
	void join_bystr(const std::vector<std::string> &vec, std::string &str, const std::string &sep);

	void simpleTokenize(const std::string & input,
		std::vector<std::string> & output,
		const std::string & separators);

	bool emptyLine(const std::string & input);

	double toDouble(const std::string &s);
	int toInteger(const std::string & s);

	void toString(const int i, std::string& s);

	std::string toUpper(const std::string & s);

	std::string toLower(const std::string & s);

	bool startsWith(const std::string & big,
		const std::string & small);

	bool endsWith(const std::string & big,
		const std::string & small);

	std::string stripString(const std::string & s,
		int left,
		int right);

	//bool my_getline(std::ifstream &inf, std::string &line);

	bool contain_uppercase_character(const std::string & s);
	bool contain_hyphen(const std::string & s);
	bool contain_number(const std::string & s);

	int getCharactersFromUTF8String(const std::string &s, std::vector<std::string> &chars);
}

#endif /* EGSTRA_CHAR_UTILS_H */
