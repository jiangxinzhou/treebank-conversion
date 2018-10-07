/* GzFile : a class with some utilities for opening (optionally)
gzipped files

Author : Terry Koo
maestro@csail.mit.edu */
#ifndef EGSTRA_GZFILE_H
#define EGSTRA_GZFILE_H

#include <stdio.h>

namespace egstra {
	class gzfile {
	public:
		static FILE* gzopen(const char* const fn, const char* const mode);
		static void gzclose(const char* const fn, FILE* const f);
	};
}

#endif /* EGSTRA_GZFILE_H */
