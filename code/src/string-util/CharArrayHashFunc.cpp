/**
* Implements the hash function for the string hash map
* @file CharArrayHashFunc.cc
* @author Mihai Surdeanu
*/

#include "CharArrayHashFunc.h"
#include <iostream>

namespace egstra {

	size_t CharArrayHashFunc::operator()(const char * s) const
	{
		int hashTemp = 0;

		for(unsigned int i = 0; s[i] != 0; i ++){
			if(0 > hashTemp) hashTemp = (hashTemp << 1) + 1;
			else hashTemp = hashTemp << 1;
			hashTemp ^= s[i];
		}

		return (size_t(hashTemp));
	}


	/*	// Huang Yun@Singapore 's hasher
	size_t CharArrayHashFunc::operator()( const char *s ) const
	{
		const size_t len = strlen(s);
		static std::vector<size_t> vSeed;
		//  initialize vSeed
		if (vSeed.empty())
		{
			std::cerr << "\n--initialing Huang Yun@Singapore 's hasher, use a lot of rand()--" << std::endl;
			vSeed.reserve(256 * 256);
			for (size_t i=0; i<vSeed.capacity(); ++i)
			{
				size_t n = 0;
				for (size_t j=0; j<sizeof(size_t); ++j)
				{
					unsigned char uc = rand() % 256;
					n <<= 8;
					n += uc;
				}
				vSeed.push_back(n);
			}
		}

		//  hash
		size_t n = 0;
		for (size_t i=0; i<len; ++i)
		{
			size_t index = (i * 256 + static_cast<size_t>(s[i]) % 256) % vSeed.size();
			n ^= vSeed[index];
		}

		return n;
	}
*/
}
