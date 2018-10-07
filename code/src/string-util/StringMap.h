/**
* Implements a generic hash map with string keys
* @file StringMap.h
* @author Mihai Surdeanu
*/

#ifdef _WIN32
#include <hash_map>
#else
#include <ext/hash_map>
#endif


#include <list>

#include <stdlib.h>

#include "HashMap.h"
#include "CharArrayEqualFunc.h"
#include "CharArrayHashFunc.h"

#ifndef EGSTRA_STRING_MAP_H
#define EGSTRA_STRING_MAP_H

namespace egstra {

	template <class T>
	class StringMap {

	public:

//#ifdef _WIN32
//		typedef HashMap<const char *, T, egstra::CharArrayHashFunc> _internal_map_t;
//#else
//		typedef HashMap<const char *, T, egstra::CharArrayHashFunc, egstra::CharArrayEqualFunc> _internal_map_t;
//#endif
#ifdef _WIN32
		typedef stdext::hash_map<const char *, T, egstra::CharArrayHashFunc> _internal_map_t;
#else
		typedef __gnu_cxx::hash_map<const char *, T, egstra::CharArrayHashFunc, egstra::CharArrayEqualFunc> _internal_map_t;
#endif

		typedef typename _internal_map_t::iterator iterator;
		typedef typename _internal_map_t::const_iterator const_iterator;

		StringMap() {
		}

		~StringMap() {
			clear();
		}

		void clear() {
			/* it's important to not deallocate the string key until after
			the it != end() comparison and ++it operations, as those
			operations may involve use of the string keys.  therefore, we
			maintain a backlog of one string. */
			const char* last = NULL;
			for(iterator it = _map.begin(); it != _map.end(); ++it) {
				if(last != NULL) { free((void*)last); }
				last = it->first;
			}
			if(last != NULL) { free((void*)last); }
			_map.clear();
		}

		bool set(const char * key, const T & value) {
			if(contains(key)){
				return false;
			}

			int keyLen = 0;
			for(; key[keyLen] != 0; keyLen ++);

			char * newKey = (char*) malloc((keyLen + 1)*sizeof(char));
			for(int i = 0; i < keyLen; i ++) { newKey[i] = key[i]; }
			newKey[keyLen] = '\0';

			_map[newKey] = value;

			return true;
		}

		bool overwrite(const char * key, const T & value) {
			if(contains(key)){
				iterator it = _map.find(key);
				it->second = value;
				return true;
			} else {
				return set(key, value);
			}
		}

		bool get(const char * key, T & value) const {
			const_iterator it;

			if((it = _map.find(key)) != _map.end()){
				value = (* it).second;
				return true;
			}

			return false;
		}

		T* get(const char * key) {
			iterator it = _map.find(key);

			if(it != _map.end()){
				return &(it->second);
			}

			return NULL;
		}

		bool contains(const char * key) const {
			if(_map.find(key) != _map.end()){
				return true;
			}

			return false;
		}

		//     void getKeys(std::vector<std::string> & keys) const {
		//       for(const_iterator it = begin(); it != end(); it ++){
		// 	keys.push_back((* it).second->getKey());
		//       }
		//     }

		size_t size() const { return _map.size(); };

		bool empty() const { return _map.empty(); };

		const_iterator begin() const { return _map.begin(); };

		const_iterator end() const { return _map.end(); };

		iterator mbegin() { return _map.begin(); };

		iterator mend() { return _map.end(); };

	protected:

		_internal_map_t _map;

	};

	void loadStringIntegerMap(StringMap<int> & map,
		const std::string & fileName);
}

#endif /* EGSTRA_STRING_MAP_H */
