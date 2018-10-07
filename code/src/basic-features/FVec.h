/* FVec : structure representing a sparse primal feature vector

Author : Terry Koo
maestro@csail.mit.edu */

#include "CppAssert.h"

#ifndef EGSTRA_FVEC_H
#define EGSTRA_FVEC_H


namespace egstra {
	class fvec {
	public:
		fvec() : idx(0), val(0), n(-1), offset(0) {}
		fvec(const fvec &rhs) : idx(rhs.idx), val(rhs.val), n(rhs.n), offset(rhs.offset) {}
		fvec & operator=(const fvec &rhs) {
			idx = rhs.idx;
			val = rhs.val;
			n = rhs.n;
			offset = rhs.offset;
			return *this;
		}

		void dealloc() {
			if (idx) {
				assert(n > 0);
				delete [] idx;
				idx = 0;
			}
			if (val) {
				assert(n > 0);
				delete [] val;
				val = 0;
			}
			n = -1;
		}

		~fvec() {} // do NOT delete *idx and *val! The user need to do it explicitly!

	public:
		const int* idx;    /* list of feature indices */
		const double* val; /* list of feature values.  if this is NULL,
						   then indicator features are assumed */
		int n;             /* number of features */
		int offset;        /* a base offset in the feature space.  each
						   feature index in idx[] is interpreted as
						   (idx[k] + offset). */

		// append elements in <rhs>, deep copy
		fvec &append(const fvec &rhs) {
			if (!rhs.idx) return *this;
			assert(n >= 0);
			assert(rhs.n > 0);
			assert(offset == rhs.offset);
			if (idx) assert (n > 0);

			const int new_n = n + rhs.n;
			int * const new_idx = new int[new_n];
			double * const new_val = (0 != val || 0 != rhs.val) ? (new double[new_n]) : 0;

			int i = 0;
			for (; i < n; ++i) {
				new_idx[i] = idx[i];
				new_val[i] = val ? val[i] : 1.;
			}
			for (; i < new_n; ++i) {
				new_idx[i] = rhs.idx[i-n];
				if (new_val) new_val[i] = rhs.val ? rhs.val[i-n] : 1.;
			}

			if (idx) delete [] idx;
			if (val) delete [] val;
			idx = new_idx;
			val = new_val;
			n = new_n;

			return *this;
		}
	// append elements in <rhs>, deep copy
		// use the same value for all elements in <rhs>		
		fvec &append(const fvec &rhs, double rhs_val) {
			if (!rhs.idx) return *this;
			assert(n >= 0);
			assert(rhs.n > 0);
			assert(offset == rhs.offset);
			if (idx) assert (n > 0);

			const int new_n = n + rhs.n;
			int * const new_idx = new int[new_n];
			double * const new_val = new double[new_n];

			int i = 0;
			for (; i < n; ++i) {
				new_idx[i] = idx[i];
				new_val[i] = val ? val[i] : 1.;
			}
			for (; i < new_n; ++i) {
				new_idx[i] = rhs.idx[i-n];
				new_val[i] = rhs_val;
			}

			if (idx) delete [] idx;
			if (val) delete [] val;
			idx = new_idx;
			val = new_val;
			n = new_n;

			return *this;
		}	
};
}

#endif /* EGSTRA_FVEC_H */


