/**
* Implements the equals method for the string hash map
* @file CharArrayEqualFunc.h
* @author Mihai Surdeanu
*/

#ifndef EGSTRA_CHAR_ARRAY_EQUAL_FUNC_H
#define EGSTRA_CHAR_ARRAY_EQUAL_FUNC_H

namespace egstra {

	typedef struct CharArrayEqualFunc
	{
		bool operator()(const char * s1, const char * s2) const;
	} 
	CharArrayEqualFunc;

}

#endif /* EGSTRA_CHAR_ARRAY_EQUAL_FUNC_H */
