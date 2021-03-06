/* FVec : structure representing a sparse primal feature vector

Author : Terry Koo
maestro@csail.mit.edu */

#include "CppAssert.h"

#ifndef EGSTRA_FVEC_H
#define EGSTRA_FVEC_H

namespace egstra {
	class fvec_base {
	public:
		int cnt;   // how many pointer is referring to this vector *duplicate usage of one single fvec*
		const int* idx;    /* list of feature indices */
		const double* val; /* list of feature values.  if this is NULL,
						   then indicator features are assumed */
		int n;             /* number of features */

		/*****  dealloc() and getpointer() must used in pairs *****/
		void dealloc() {
			if (--cnt >= 0) return;
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
		const fvec_base *const getpointer() {
			++cnt;
			return this;
		}
		/*****  dealloc() and getpointer() must used in pairs *****/
		
		// DO NOT USE: fvec_base fv_base;
		// USE instead: new fvec_base();
		fvec_base() : cnt(0), idx(0), val(0), n(-1) {}
		~fvec_base() {
			if (cnt != 0 || idx != 0 || val != 0 || n != -1) {
				cerr << "~fvec_base(): cnt != 0 || idx != 0 || val != 0 || n != -1" << endl;
				exit(-1);
			}
		} 
	private:
		fvec_base(const fvec_base &rhs) : cnt(0), idx(rhs.idx), val(rhs.val), n(rhs.n) { 
			cerr << "fvec_base(const fvec_base &rhs) prohibited!\n"; exit(-1); 
		}
		fvec_base & operator=(const fvec_base &rhs) {
			cerr << "fvec_base(const fvec_base &rhs) prohibited!\n"; exit(-1); 
			cnt = rhs.cnt;
			idx = rhs.idx;
			val = rhs.val;
			n = rhs.n;
			return *this;
		}
	}

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



		~fvec() {} // do NOT delete *idx and *val! The user need to do it explicitly!

	public:

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


