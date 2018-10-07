#include <stdio.h>
#include <assert.h>
#include <iostream>

#include "Parameters.h"

#define myname "Parameters"

using namespace std;

namespace egstra {

	parameters::parameters(const int dim)
		: _dim(dim), _W(NULL), _Wsum(0)/*, _Waver(0)*/, _Wtime(0)
	{
		realloc(_dim);
	}

	parameters::parameters(double* const V, const int dim)
		: _dim(dim), _W(V), _Wsum(0)/*, _Waver(0)*/, _Wtime(0)
	{
		assert(_dim > 0);
		realloc(_dim);
		/* set the proper squared-norm */
		for(int i = 0; i < _dim; ++i) {
			const double val = _W[i];
		}
	}

	parameters::~parameters() { dealloc(); }


	void parameters::dealloc() {
		if(_W != NULL) { 
			delete [] _W;
			//delete [] _Waver;
			delete [] _Wsum;
			delete [] _Wtime;
		}
	}

	void parameters::realloc(const int dim) {
		dealloc();
		_dim = dim;
		assert(_dim >= 0);

		if(dim > 0) {
			_W = new double[_dim];
			//_Waver = new double[_dim];
			_Wsum = new double[_dim];
			_Wtime = new int[_dim];
			assert(_W != NULL);
			//assert(_Waver != NULL);
			assert(_Wsum != NULL);
			assert(_Wtime != NULL);
			zero();
		} else {
			_W = NULL;
			_Wsum = 0;
			//_Waver = 0;
			_Wtime = 0;
		}
	}

	void parameters::save(const string& dir, const int t, const bool w_sum,
		const string& stem) const {
			char* const fname = new char[64 + stem.size()];
			sprintf(fname, "%s%s.%03d.gz", (w_sum ? "sum." : ""), stem.c_str(), t);
			char* const command = new char[64 + stem.size() + dir.size()];

/*			sprintf(command, "gzip -c > '%s/%s'", dir.c_str(), fname);
			cerr << myname << " : writing to \""
				<< dir << "/" << fname << "\" " << flush;

		FILE* gzout = popen(command, "w");
            while (gzout == NULL) {
                cerr << "error popen: " << command << endl;
                cerr << "sleep for 60 seconds..." << endl;
                sleep(60);
			    gzout = popen(command, "w");
            }
*/
			sprintf(command, "%s/%s", dir.c_str(), fname);
            cerr << "\nsave parameters to \"" << command << "\" " << endl;

            FILE *gzout = fopen(command, "w");
            assert(gzout != NULL);
			fprintf(gzout, "%d\n", w_sum ? _dim : _Wtime[0]); 

			delete [] fname;
			delete [] command;

			const double * const p = w_sum ? _Wsum : _W;
			/* only print nonzero parameters */
			int nnz = 0;
			for (int i = 0; i < _dim; ++i) {
				if(p[i] >= dparser::ZERO || p[i] <= -dparser::ZERO) {
					fprintf(gzout, "%d %.16g\n", i, p[i]);
					if(((++nnz) & 0xfffff) == 0) { cerr << "." << flush; }
				}
			}

			/* assert(pclose(gzout) != -1); */
			fclose(gzout);
			cerr << "\nparameters::save [nnz=" << nnz << "/" << _dim << "]" << endl;
	}

	void parameters::delete_file(const string& dir, const int t,
		const string& stem) {
#ifndef _WIN32
			char* const cmd = new char[dir.size() + 64 + stem.size()];
			sprintf(cmd, "rm %s/%s%s.%03d.gz", dir.c_str(), "", stem.c_str(), t);
            system(cmd);
			sprintf(cmd, "rm %s/%s%s.%03d.gz", dir.c_str(), "sum.", stem.c_str(), t);
            system(cmd);
#endif
    }

	void parameters::load(const string& dir, const int t, const bool w_sum,
		const string& stem) {
			char* const fname = new char[64 + stem.size()];
			sprintf(fname, "%s%s.%03d.gz", w_sum ? "sum." : "", stem.c_str(), t);
			char* const command = new char[64 + stem.size() + dir.size()];
/*			sprintf(command, "gunzip -c '%s/%s'", dir.c_str(), fname);

			FILE* gzin = popen(command, "r");
            while (gzin == NULL) {
                cerr << "error popen: " << command << endl;
                cerr << "sleep for 60 seconds..." << endl;
                sleep(60);
			    gzin = popen(command, "r");
            }*/
			sprintf(command, "%s/%s", dir.c_str(), fname);
			cerr << myname << " : reading from \""
				<< command << "\" " << flush;

            FILE *gzin = fopen(command, "r");
			assert(gzin != NULL);

			int tmp;
			assert(fscanf(gzin, "%d\n", &tmp) == 1);
			assert(tmp > 0);
			if (w_sum) {
				//_dim = (dim > 0 ? dim : tmp);
				//realloc(_dim);
			} else {
				set_time(tmp);
			}

			delete [] fname;
			delete [] command;

			double * const p = w_sum ? _Wsum : _W;
			/* the parameters file can omit zero values, so we need to read
			lines on-demand */
			int index = -1;
			double value = 0;
			int nnz = 0;
			while(1) {
				const int nread = fscanf(gzin, "%d %lf\n", &index, &value);
				if(nread <= 0 && feof(gzin)) { break; }
				assert(nread == 2);
				assert(index >= 0);
				assert(index < _dim);
				p[index] = value;
				if(((++nnz) & 0xfffff) == 0) { cerr << "." << flush; }
			}

			/* assert(pclose(gzin) != -1); */
			fclose(gzin);
			cerr << " [nnz=" << nnz << "/" << _dim << "]" << endl;
	}
}

#undef myname


