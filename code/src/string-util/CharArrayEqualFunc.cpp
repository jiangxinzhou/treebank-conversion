/**
 * Implements the equals method for the string hash map
 * @file CharArrayEqualFunc.cc
 * @author Mihai Surdeanu
 */

#include <string.h>

#include "CharArrayEqualFunc.h"

namespace egstra {
  bool CharArrayEqualFunc::operator()(const char * s1,
				      const char * s2) const
  {
    return (strcmp(s1, s2) == 0);
  }
}
