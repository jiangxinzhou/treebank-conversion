/**
* Implements the feature dictionary
* @file FeatureDictionary.cc
* @author Mihai Surdeanu
*/

#include <iostream>
#include <iomanip>
#include <fstream>

#include <string.h>
#include <stdio.h>

#include "FeatureDictionary.h"
#include "CharUtils.h"
#include "Constants.h"
#include "CppAssert.h"
#include "GzFile.h"

using namespace std;

namespace egstra {

	int FeatureDictionary::getFeature(const std::string & name,
		bool create)
	{
		IndexAndFrequency* iaf;

		// this feature already exists
		if((iaf = mMap.get(name.c_str())) != NULL) {
			// increment the feature frequency
			if(create) iaf->mFreq ++;
			return iaf->mIndex;
		}

		// the feature does not exist
		if(create == false) return -1;

		// must create a new feature
		// int index = mMap.size();
		++mMaxDim;
		int index = mMaxDim;
		mMap.set(name.c_str(), IndexAndFrequency(index));
		//cerr << "Created feature \"" + name + "\" with index: " << index << "\n";
		return index;
	}

	void FeatureDictionary::print_dic(){
		for(StringMap<IndexAndFrequency>::const_iterator it = mMap.begin();
			it != mMap.end();
			++it) {
			cerr << it->first << endl;
				/*const string& key = it->first;
				IndexAndFrequency iaf;
				mMap.get(key.c_str(), iaf);
				cerr << key.c_str() << "\t" << iaf.mFreq << "\t" << iaf.mIndex << endl;*/
		}
	}

	void FeatureDictionary::load(const std::string & fileName,
		const int cutoff) {
			// reset the previous content
			mMap.clear();

			//FILE* const f = fopen(fileName.c_str(), "r");
			FILE* const f = gzfile::gzopen(fileName.c_str(), "r");
			char* const buf = new char[16384];
			int idx = 0;
			int cnt = 0;
			int line = 0;

			cerr << "FeatureDictionary : loading from \"" << fileName
				<< "\" " << fixed << setprecision(1) << endl;
			cerr << "FeatureDictionary : " << flush;
			while(1) {
				const int nread = fscanf(f, "%16383s %d", buf, &cnt);
				if(nread <= 0 && feof(f)) { break; }
				else                      { assert(nread == 2); }

				if(cnt >= cutoff) {
					mMap.set(buf, IndexAndFrequency(idx, cnt));
					//cerr << buf << ' ' << idx << ' ' << cnt << endl;
					++idx;
				}
				++line;

				if((line & 0x1fffff) == 0) {
					cerr << "(" << (double)idx/1000000.0 << "m)" << flush;
				} else if((line & 0x3ffff) == 0) {
					cerr << "." << flush;
				}
			}
			delete [] buf;
			//fclose(f);
			gzfile::gzclose(fileName.c_str(), f);
			cerr << " " << idx << " features";
			if(cutoff > 1) { cerr << " (" << line << " total)"; }
			cerr << endl;

			mMaxDim = idx - 1;
	}

	void FeatureDictionary::save(const std::string & fileName) /*const*/
	{
		//FILE* const f = fopen(fileName.c_str(), "w");
		FILE* const f = gzfile::gzopen(fileName.c_str(), "w");
		int count = 0;

		cerr << "FeatureDictionary : saving to \"" << fileName
			<< "\" " << fixed << setprecision(1) << endl;
		cerr << "FeatureDictionary : " << flush;
		for(StringMap<IndexAndFrequency>::const_iterator it = mMap.begin();
			it != mMap.end();
			++it) {
				const string& key = it->first;
				if(key == "-NULL-")
					continue;
				IndexAndFrequency iaf;
				mMap.get(key.c_str(), iaf);
				fprintf(f, "%s %d\n", key.c_str(), iaf.mFreq);
				count ++;

				if((count & 0x1fffff) == 0) {
					cerr << "(" << (double)count/1000000.0 << "m)" << flush;
				} else if((count & 0x3ffff) == 0) {
					cerr << "." << flush;
				}
		}
		//fprintf(f, "%s %d\n", "-NULL-", 1);

		//fclose(f);
		gzfile::gzclose(fileName.c_str(), f);
		cerr << " " << count << " features" << endl;

		clear();
	}

	typedef pair<string, int> Pair;
	int cmp_pair(const Pair &cmp1, const Pair &cmp2){
		return cmp1.second > cmp2.second;
	}

	void FeatureDictionary::saveAfterSort(const std::string & fileName, int minCount) /*const*/
	{
		//FILE* const f = fopen(fileName.c_str(), "w");
		FILE* const f = gzfile::gzopen(fileName.c_str(), "w");
		int count = 0;

		cerr << "FeatureDictionary : saving to \"" << fileName
			<< "\" " << fixed << setprecision(1) << endl;
		cerr << "FeatureDictionary : " << flush;
		vector<Pair> sort_vect;
		for (StringMap<IndexAndFrequency>::const_iterator it = mMap.begin();it != mMap.end();++it) {
			const string& key = it->first;
			IndexAndFrequency iaf;
			mMap.get(key.c_str(), iaf);
			sort_vect.push_back(make_pair(key, iaf.mFreq));
		}
		sort(sort_vect.begin(), sort_vect.end(), cmp_pair);
		//for (vector<Pair> ::iterator it = sort_vect.begin(); it != sort_vect.end(); it++){
		for(int i=0;i<sort_vect.size();i++){
			if (sort_vect[i].second <= minCount)
				break;
			fprintf(f, "%s %d\n", sort_vect[i].first.c_str(), sort_vect[i].second);
			
			count++;

			if ((count & 0x1fffff) == 0) {
				cerr << "(" << (double)count / 1000000.0 << "m)" << flush;
			}
			else if ((count & 0x3ffff) == 0) {
				cerr << "." << flush;
			}
		}
		IndexAndFrequency iaf;
		mMap.get("-UNKNOWN-", iaf);
		if (iaf.mFreq <= minCount)
			fprintf(f, "%s %d\n", "-UNKNOWN-", iaf.mFreq);
		//fclose(f);
		gzfile::gzclose(fileName.c_str(), f);
		cerr << " " << count << " features" << endl;

		clear();
	}
	
	void FeatureDictionary::map_all(vector<int>& fidx,
		const list<string>& fstr,
		const bool create) {
			fidx.reserve(fidx.size() + fstr.size());
			list<string>::const_iterator it = fstr.begin();
			const list<string>::const_iterator it_end = fstr.end();
			for(; it != it_end; ++it) {
				const int idx = getFeature(*it, create);
				assert(idx >= -1);
				if(idx >= 0) { fidx.push_back(idx); }
			}
	}

	int FeatureDictionary::map_all(int* const fidx,
		const list<string>& fstr,
		const bool create) {
			list<string>::const_iterator it = fstr.begin();
			const list<string>::const_iterator it_end = fstr.end();
			int n = 0;
			for(; it != it_end; ++it) {
				const int idx = getFeature(*it, create);
				assert(idx >= -1);
				if(idx >= 0) { fidx[n] = idx; ++n; }
			}
			return n;
	}

	void FeatureDictionary::map_all( fvec * const fv, 
		const int offset,		
		const std::list<std::string>& fstr, 
		const bool create )
	{
		assert(offset >= 0);
		assert(fv->idx == 0);

		vector<int> fidx;
		map_all(fidx, fstr, create);

		fv->idx = 0;
		fv->val = 0;
		fv->offset = offset;
		fv->n = fidx.size();

		if (!fidx.empty()) {
			int * const f = new int[fv->n];
			for (int i = 0; i < fv->n; ++i) f[i] = fidx[i];
			fv->idx = f;
		}
	}

	void FeatureDictionary::map_all(vector<int>& fidx,
		std::vector<double>& val_left,
		const list<string>& fstr,
		const std::list<double>& val, 
		const bool create) {
			assert(fidx.empty() && val_left.empty());
			assert(fidx.size() >= val_left.size());
			assert(fstr.size() == val.size());
			if (fidx.size() != val_left.size()) {
				val_left.resize(fidx.size(), 1.0);
			}

			fidx.reserve(fidx.size() + fstr.size());
			val_left.reserve(fidx.size() + fstr.size());

			list<string>::const_iterator it = fstr.begin();
			list<double>::const_iterator it2 = val.begin();
			const list<string>::const_iterator it_end = fstr.end();
			for(; it != it_end; ++it, ++it2) {
				const int idx = getFeature(*it, create);
				assert(idx >= -1);
				if(idx >= 0) { 
					fidx.push_back(idx); 
					val_left.push_back(*it2);
				}
			}
	}

	void FeatureDictionary::map_all( fvec * const fv, 
		const int offset, 
		const std::list<std::string>& fstr, 
		const std::list<double>& val, 
		const bool create )
	{
		assert(offset >= 0);
		assert(fv->idx == 0);

		vector<int> fidx;
		vector<double> val_left;
		map_all(fidx, val_left, fstr, val, create);

		fv->idx = 0;
		fv->val = 0;
		fv->offset = offset;
		fv->n = fidx.size();

		if (!fidx.empty()) {
			int * const f = new int[fv->n];
			double * const v = new double[fv->n];
			for (int i = 0; i < fv->n; ++i) {
				f[i] = fidx[i];
				v[i] = val_left[i];
			}
			fv->idx = f;
			fv->val = v;
		}
	}
	

	void FeatureDictionary::collect_keys( const char ** const keys, const int sz ) const
	{
		assert(sz == dimensionality());
		for( StringMap<IndexAndFrequency>::const_iterator it = mMap.begin(); it != mMap.end(); ++it) {
			const char *key = it->first;
			const int idx = it->second.mIndex;
			assert(idx >= 0 && idx < sz);
			assert(NULL == keys[idx]);
			keys[idx] = key;
		}
	}
}

