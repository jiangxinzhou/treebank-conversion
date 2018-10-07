/**
* Implements a generic hash map
* @file HashMap.h
* @author Mihai Surdeanu
*/
#ifdef _WIN32
#include <hash_map>
#else
#include <ext/hash_map>
#endif

#ifndef EGSTRA_HASH_MAP_H
#define EGSTRA_HASH_MAP_H

namespace egstra {
#ifdef _WIN32
	template <class K, class V, class F>
	class HashMap : public stdext::hash_map<K, V, F> {};
#else
	template <class K, class V, class F, class E>
	class HashMap : public __gnu_cxx::hash_map<K, V, F, E> {};
#endif
}

#endif /* EGSTRA_HASH_MAP_H */
