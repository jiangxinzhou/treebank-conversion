#include "StringMap.h"

#include "Constants.h"
#include "CharUtils.h"
#include "CppAssert.h"

#include <fstream>
#include <vector>
#include <string>

using namespace std;

namespace egstra {
	void loadStringIntegerMap(StringMap<int>& map,
		const string & fileName)
	{
		std::ifstream is(fileName.c_str());
		//RVASSERT(is, "Can not open file: " << fileName);
		char line[MAX_FEATDICT_LINE];

		while(is.getline(line, MAX_FEATDICT_LINE)){
			vector<string> tokens;
			simpleTokenize(line, tokens, " \t");
			RVASSERT(tokens.size() == 2,
				"Invalid line \"" << line << "\" in file: " << fileName);

			map.set(tokens[1].c_str(), toInteger(tokens[0]));
		}

		is.close();
	}
}
