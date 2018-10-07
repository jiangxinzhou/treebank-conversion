
/************************************************************
*
*   Author : Terry Koo
*            maestro@csail.mit.edu
*
*   Parameter vector class
*
************************************************************/
#ifndef EGSTRA_PARAMETERS_H
#define EGSTRA_PARAMETERS_H

//#include <ext/hash_map>  /* for sparse vectors */
#include <string>
#include <iostream>

#ifdef _WIN32
#include <hash_map>
#else
#include <ext/hash_map>
#endif


#include <assert.h>
#include <math.h>
#include "FVec.h"
#include "CppAssert.h"

#include "common.h"

/* cutoff point at which the parameters should be recentered */
#define EGSTRA_PARAMETERS_SCALE_CUTOFF   0.1

namespace egstra {

	/* sparse feature/parameter vector */

#ifdef _WIN32
	typedef stdext::hash_map<int,double> sparsevec;
#else
	typedef __gnu_cxx::hash_map<int, double> sparsevec;
#endif

	/* the parameters are stored internally as c-style array.  the
	_alpha and _Wsq values allow a fast implementation of certain
	operations. */
	class parameters {
	private:
		int _dim;         /* dimensionality */
		double* _W;       /* parameter vector */
//		double* _Waver;		/* averaged parameter vector */
		double* _Wsum;	/* sum over primal parameters */
		int* _Wtime;	/* last-update timestamps for each parameter */

		/* release the arrays */

	public:

		/**** lifecycle **********************************************/
		/* create zeroed parameters */
		parameters(const int dim = 0);
		/* create a wrapper around the given vector */
		parameters(double* const V, const int dim);
		/* destroy and delete memory (if _self_alloc) */
		~parameters();

		/* store/read to file */
		void delete_file(const std::string& fname, const int t,
			const std::string& stem = "parameters");
		void save(const std::string& fname, const int t, const bool w_sum = false,
			const std::string& stem = "parameters") const;
		void load(const std::string& fname, const int t, const bool w_sum = false,
			const std::string& stem = "parameters");

		/* resize the arrays */
		void realloc(const int dim);
		void dealloc();

		int get_time() const {
			return _Wtime[0];
		}

		/* accessors */
		int dim() const { return _dim; }

		/* set a batch of values via a sparse vector *
		void set_values(const sparsevec& sV) {
			sparsevec::const_iterator sV_i = sV.begin();
			const sparsevec::const_iterator sV_end = sV.end();
			for(; sV_i != sV_end; ++sV_i) {
				const int idx = sV_i->first;
				assert(idx >= 0);
				assert(idx < _dim);
				_W[idx] = sV_i->second;
			}
		} */

		/* perform all lazy updates on the sum-vector */
		void flush_avg(const int now) {
			for(int i = 0; i < dim(); ++i) {
				_Wsum[i] += (now - _Wtime[i]) * _W[i];
				_Wtime[i] = now;
			}
		}

		//void average() {
		//	for(int i = 0; i < dim(); ++i)
		//		_Waver[i] = _Wsum[i] / _Wtime[0];
		//}

		/**** Read-only computations **********************************/
		/* return the inner product W . F */
		double dot(const fvec* const F, const bool use_aver = false) const {
			const double * const p = use_aver ? _Wsum : _W;
			const int n = F->n;
			assert(n >= 0);
			const int o = F->offset;
			const int* const fidx = F->idx;
			const double* const fval = F->val;
			double iprod = 0;
			if(fval == NULL) { /* indicators */
				for(int i = 0; i < n; ++i) {
					const int idx = o + fidx[i];
					assert(idx >= 0);
					assert(idx < _dim);
					iprod += p[idx];
				}
			} else {
				for(int i = 0; i < n; ++i) {
					const int idx = o + fidx[i];
					assert(idx >= 0);
					assert(idx < _dim);
					iprod += p[idx]*fval[i];
				}
			}

			return iprod;
		}

		/* return the inner product of W with a batch of feature vectors *
		void dot(double* const S, const fvec* const F, const int R) const {
			for(int r = 0; r < R; ++r) { S[r] = dot(F + r); }
		}
		/* return the inner product of W with a batch of feature vectors *
		void dot(double* const S, const fvec* const F, const int R,
			const double scale) const {
				for(int r = 0; r < R; ++r) { S[r] = scale*dot(F + r); }
		}
		*/

		/* return the inner product W . U */
		double dot(const sparsevec& U, const bool use_aver = false) const {
			const double * const p = use_aver ? _Wsum : _W;
			double iprod = 0;
			sparsevec::const_iterator ui = U.begin();
			const sparsevec::const_iterator uend = U.end();
			for(; ui != uend; ++ui) {
				const int idx = ui->first;
				assert(idx >= 0);
				assert(idx < _dim);
				const double val = ui->second;
				iprod += val * p[idx];
			}
			return iprod;
		}

		/**** Modifying computations **********************************/
		/* perform W = 0 */
		void zero() {
			for(int i = 0; i < _dim; ++i) { 
				_W[i] = 0.0;
				_Wsum[i] = 0.0;
				//_Waver[i] = 0.0;
				_Wtime[i] = 0;
			}
		}

		void set_time(const int now) {
			for(int i = 0; i < _dim; ++i) {
				_Wtime[i] = now;
			}
		}

		/* perform W[i] += d *
		void add(const int i, const double d) {
			assert(i >= 0);
			assert(i < _dim);
			_W[i] += d;
		} */

		/* perform W += scale * F *
		void add(const fvec* const F, const double scale = 1.0) {
			const int n = F->n;
			const int o = F->offset;
			const int* const fidx = F->idx;
			const double* const fval = F->val;
			if(fval == NULL) { /* indicators *
				const double updsq = scale*scale;
				for(int i = 0; i < n; ++i) {
					const int idx = o + fidx[i];
					RVASSERT(idx >= 0, "Invalid feat index: " << idx);
					if (idx < 0) {
						throw std::string("bad idx < 0");
						std::cerr << "bad idx < 0" << std::endl;
					}

					assert(idx >= 0);
					assert(idx < _dim);
					const double cur_val = _W[idx];
					_W[idx] = cur_val + scale;
				}
			} else {
				for(int i = 0; i < n; ++i) {
					const int idx = o + fidx[i];
					RVASSERT(idx >= 0, "Invalid feat index: " << idx);
					if (idx < 0) {
						throw std::string("bad idx < 0");
						std::cerr << "bad idx < 0" << std::endl;
					}
					assert(idx >= 0);
					assert(idx < _dim);
					const double upd = scale * fval[i];
					const double cur_val = _W[idx];
					_W[idx] = cur_val + upd;
				}
			}
		} */

		/* perform W[k] += scale * U 
			also change _Wsum, _Wtime. */
		void add(const sparsevec& U, const int now, const double scale = 1.0) {
			sparsevec::const_iterator ui = U.begin();
			const sparsevec::const_iterator uend = U.end();
			for(; ui != uend; ++ui) {
				const int idx = ui->first;
				assert(idx >= 0);
				assert(idx < _dim);
				const double upd = scale * ui->second;
				const double cur_val = _W[idx];
				_W[idx] = cur_val + upd;

				// consider _Wsum & _Wtime
				// elapsed == 0 means updating the same weight again for the same instance

				// NOTE: _Wsum DOES include _W after update, which is for the convenient process of _Wsum 
				//			and is different from the implementation in EGSTRA.
				const int elapsed = now - _Wtime[idx];
				_Wsum[idx] += elapsed * cur_val + upd;
				_Wtime[idx] = now;
			}
		}

		/**** Static utilities **********************************/
		/* find V^2 for sparse vectors */
		static double sparse_sqL2(const sparsevec& V) {
			double norm = 0;
			sparsevec::const_iterator V_i = V.begin();
			const sparsevec::const_iterator V_end = V.end();
			for(; V_i != V_end; ++V_i) {
				const double val = V_i->second;
				norm += val*val;
			}
			return norm;
		}

		/* set V += scale * F for sparse vectors */
		static void sparse_add(sparsevec& V,
			const fvec* const F,
			const double scale = 1.0) {
				const int n = F->n;
				const int o = F->offset;
				const int* const fidx = F->idx;
				const double* const fval = F->val;
				if(fval == NULL) { /* indicators */
					for(int i = 0; i < n; ++i) {
						const int idx = o + fidx[i];
						if (V.find(idx) == V.end()) V[idx] = 0; // added by zhenghua (just for sure)
						V[idx] += scale; // zhenghua: what if V[idx] dost not exist??
					}
				} else {
					for(int i = 0; i < n; ++i) {
						const int idx = o + fidx[i];
						const double update = scale*(fval[i]);
						if (V.find(idx) == V.end()) V[idx] = 0; 
						V[idx] += update;
					}
				}
		} 

		/* set V += scale * F for sparse vectors */
		static void sparse_add(sparsevec& V,
			const sparsevec &add_me,
			const double scale = 1.0) {
			    sparsevec::const_iterator ui = add_me.begin();
			    const sparsevec::const_iterator uend = add_me.end();
		    	for(; ui != uend; ++ui) {
		    		const int idx = ui->first;
                    if (V.find(idx) == V.end()) V[idx] = 0;
                    V[idx] += scale * ui->second;
			    }
		} 

		/* array access: these involve multiplication and division of
		doubles, so don't use them often */
		double operator[](const int i) const {
			assert(i >= 0);
			assert(i < _dim);
			return _W[i];
		}

		/* direct access to the parameter vector.  use only if you know
		what you're doing. */
		double* c_buf() { assert(_W != NULL); return _W; }

	};
}


#endif /* EGSTRA_PARAMETERS_H */


