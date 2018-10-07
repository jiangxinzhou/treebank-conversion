#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <ctime>
#include <list>
#include <cmath>
#include <cstring>
#include <fstream>
using namespace std;

#ifdef    _MSC_VER
	#define isfinite    _finite
#endif

#define FEAT_ID (unsigned long)
#define POSI	(unsigned int)


#define MAX_SENT_LEN 256
#define MAX_LABEL_COUNT 100
namespace dparser {
	extern const size_t CMP;
	extern const size_t INCMP;
	extern const size_t SIB_SP;
	/** Type of a float value. */
	typedef double floatval_t;

	extern const double LOG_EXP_ZERO;

	extern const double EPS;
	extern const double ZERO;
	extern const double DOUBLE_NEGATIVE_INFINITY;
	extern const double DOUBLE_POSITIVE_INFINITY;

	extern const string NO_FORM;
	extern const string FEAT_SEP;

	extern const string BET_NO_POS;
	extern const string BET_ONE_POS;

	extern const string ROOT_FORM;
	extern const int	ROOT_HEAD;

	extern int lstm_layer_num;

	extern int _g_label_num;
	extern int _g_label_num_no_null;
	extern int _g_feat_dim;
	extern int _g_feat_num;

	extern const string UNKNOWN;
	extern const string ROOT;
	extern const string NULLstr;
	extern int NULLstr_word_id;
	extern int NULLstr_pos_id;
	extern int NULLstr_label_id;

	// remove the blanks at the begin and end of string
	
	inline void clean_str(string &str) 
		{
			string blank = " \t\r\n";
			string::size_type pos1 = str.find_first_not_of(blank);
			string::size_type pos2 = str.find_last_not_of(blank);
			if (pos1 == string::npos) {
				str = "";
			} else {
				str = str.substr(pos1, pos2-pos1+1);
			}
		}
		
	inline void print_time() {

			time_t lt=time(NULL);
			string strTime = ctime(&lt);
			clean_str(strTime);
			std::cerr << "\t[" << strTime << "]" << endl;

	}

	inline void print_time(ofstream &out) {

			time_t lt=time(NULL);
			string strTime = ctime(&lt);
			clean_str(strTime);
			out << "\t[" << strTime << "]" << endl;

	}

	


	inline bool smaller_than(const double a, const double b) {
		return (a < b - EPS);
	}
	inline bool bigger_than(const double a, const double b) {
		return (a > b + EPS);
	}
	inline bool coarse_equal_to(const double a, const double b) {
		const double interval = 1e-3;
		return ( (a <= b+interval) && (a >= b-interval));
	}
	inline bool equal_to(const double a, const double b) {
		return ( (a <= b+EPS) && (a >= b-EPS));
	}
	inline bool equal_to_negative_infinite(const double a) {
		return equal_to(a, DOUBLE_NEGATIVE_INFINITY);
	}

    inline double abs(const double val) { return (val > 0 ? val : -val); }

	void get_children( const vector<int> &heads, vector< list<int> > &children_l, vector< list<int> > &children_r );
/*
	class ValueIndexPair {
	public:
		double val;
		int i1, i2;
	public:
		ValueIndexPair(double _val=0, int _i1=0, int _i2=0) : val(_val), i1(_i1), i2(_i2) {}

		int compareTo(const ValueIndexPair &other) const {
			if(val < other.val - EPS)
				return -1;
			if(val > other.val + EPS)
				return 1;
			return 0;
		}

		ValueIndexPair &operator=(const ValueIndexPair &other) {
			val = other.val; i1 = other.i1; i2 = other.i2; return *this;
		}
	};

	// Max Heap
	// We know that never more than K elements on Heap
	class BinaryHeap { 

	public:
		bool empty() {
			return currentSize == 0;
		}

		BinaryHeap(int def_cap) {
			DEFAULT_CAPACITY = def_cap;
			theArray.resize(DEFAULT_CAPACITY+1); 
			// theArray[0] serves as dummy parent for root (who is at 1) 
			// "largest" is guaranteed to be larger than all keys in heap
			theArray[0] = ValueIndexPair(DOUBLE_POSITIVE_INFINITY,-1,-1);      
			currentSize = 0; 
		} 

		BinaryHeap() {} 

		BinaryHeap &resize(int new_size) {
			DEFAULT_CAPACITY = new_size;
			theArray.resize(DEFAULT_CAPACITY+1); 
			theArray[0] = ValueIndexPair(DOUBLE_POSITIVE_INFINITY,-1,-1);      
			currentSize = 0;
			return *this;
		}

		//ValueIndexPair getMax() {
		//	return theArray[1]; 
		//}


		void add(const ValueIndexPair &e) { 
			//if (currentSize == DEFAULT_CAPACITY) {	// reach the max size
			//
			//}
			// bubble up: 
			int where = currentSize + 1; // new last place 
			while ( e.compareTo(theArray[parent(where)]) > 0 ){ 
				theArray[where] = theArray[parent(where)]; 
				where = parent(where); 
			} 
			theArray[where] = e; currentSize++;
		}

		void removeMax(ValueIndexPair &max);

	private:
		int parent(int i) { return i / 2; } 
		int getLeftChild(int i) { return 2 * i; } 
		int getRightChild(int i) { return 2 * i + 1; } 

	private:
		int DEFAULT_CAPACITY; 
		int currentSize; 
		vector<ValueIndexPair> theArray;
	};
*/
	inline void veczero(floatval_t *x, const int n)
	{
		memset(x, 0, sizeof(floatval_t) * n);
	}

	inline void vecset(floatval_t *x, const floatval_t a, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			x[i] = a;
		}
	}

	inline void veccopy(floatval_t *y, const floatval_t *x, const int n)
	{
		memcpy(y, x, sizeof(floatval_t) * n);
	}

	inline void vecadd(floatval_t *y, const floatval_t *x, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] += x[i];
		}
	}

	inline void vecaadd(floatval_t *y, const floatval_t a, const floatval_t *x, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] += a * x[i];
		}
	}

	inline void vecsub(floatval_t *y, const floatval_t *x, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] -= x[i];
		}
	}

	inline void vecasub(floatval_t *y, const floatval_t a, const floatval_t *x, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] -= a * x[i];
		}
	}

	inline void vecmul(floatval_t *y, const floatval_t *x, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] *= x[i];
		}
	}

	inline void vecinv(floatval_t *y, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] = 1. / y[i];
		}
	}

	inline void vecscale(floatval_t *y, const floatval_t a, const int n)
	{
		int i;
		for (i = 0;i < n;++i) {
			y[i] *= a;
		}
	}

	inline floatval_t vecdot(const floatval_t *x, const floatval_t *y, const int n)
	{
		int i;
		floatval_t s = 0;
		for (i = 0;i < n;++i) {
			s += x[i] * y[i];
		}
		return s;
	}

	inline floatval_t vecsum(floatval_t* x, const int n)
	{
		int i;
		floatval_t s = 0.;

		for (i = 0;i < n;++i) {
			s += x[i];
		}
		return s;
	}

	inline floatval_t vecsumlog(floatval_t* x, const int n)
	{
		int i;
		floatval_t s = 0.;
		for (i = 0;i < n;++i) {
			s += log(x[i]);
		}
		return s;
	}
}

#endif



