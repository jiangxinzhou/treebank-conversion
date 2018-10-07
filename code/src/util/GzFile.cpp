#include "GzFile.h"

#include <string.h>
#include <assert.h>
#include <stdlib.h>

namespace egstra {
	FILE* gzfile::gzopen(const char* const fn, const char* const mode) {
		FILE* f = NULL;
#ifdef _LINUX
		const int n = strlen(fn);
		if(n > 3 &&
			(fn[n - 1] == 'z' || fn[n - 1] == 'Z') &&
			(fn[n - 2] == 'g' || fn[n - 2] == 'G') &&
			fn[n - 3] == '.') {
				char* const cmd = (char*)malloc((n + 64)*sizeof(char));
				if(mode[0] == 'r') {
					snprintf(cmd, n + 64, "gunzip -c '%s'", fn);
				} else if(mode[0] == 'w') {
					snprintf(cmd, n + 64, "gzip -c > '%s'", fn);
				} else {
					printf("unrecognized mode \"%s\"\n", mode);
					exit(1);
				}
				f = popen(cmd, mode);
				free(cmd);
		} else {
			f = fopen(fn, mode);
		}
#else
		f = fopen(fn, mode);
#endif
		if(f == NULL) {
			printf("error opening \"%s\"\n", fn);
			exit(1);
		}
		return f;
	}

	void gzfile::gzclose(const char* const fn, FILE* const f) {
#ifdef _LINUX
		const int n = strlen(fn);
		if(n > 3 &&
			(fn[n - 1] == 'z' || fn[n - 1] == 'Z') &&
			(fn[n - 2] == 'g' || fn[n - 2] == 'G') &&
			fn[n - 3] == '.') {
				assert(pclose(f) != -1);
		} else {
			assert(fclose(f) == 0);
		}
#else
		assert(fclose(f) == 0);
#endif
	}
}

