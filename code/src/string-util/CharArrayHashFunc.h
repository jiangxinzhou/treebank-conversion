/**
* Implements the hash function for the string hash map
* @file CharArrayHashFunc.h
* @author Mihai Surdeanu
*/

#ifndef EGSTRA_CHAR_ARRAY_HASH_FUNC_H
#define EGSTRA_CHAR_ARRAY_HASH_FUNC_H

#include <cstdio>
#include <cstring>


#ifdef _WIN32
#include <hash_map>
#else
#include <ext/hash_map>
#endif

using namespace std;

namespace egstra {

	typedef struct CharArrayHashFunc
#ifdef _WIN32
		: public stdext::hash_compare<const char*>
#endif
	{
		size_t operator()(const char *s) const;

		bool operator() (const char* s1, const char* s2) const // compare, needed by windows-version hash_map
		{ return strcmp(s1, s2) < 0; }

	} CharArrayHashFunc;

}

#endif /* EGSTRA_CHAR_ARRAY_HASH_FUNC_H */
