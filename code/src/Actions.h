#ifndef _ACTION_H_
#define _ACTION_H_

#include "common.h"

#pragma once

// for dependency parsing, there are only four valid operations
class Actions{
public:
	enum CODE {SHIFT=0, POP_ROOT=1, LEFT_ARC=2, RIGHT_ARC=LEFT_ARC + MAX_LABEL_COUNT};
	unsigned long _code;

public:
	Actions():_code(0){
	}

	Actions(int code):_code(code) {
	}

	Actions(const Actions& ac):_code(ac._code){
	}

public:

	inline void clear() {_code = SHIFT;}

	inline void set(int code) {
		_code = code;
	}

	inline bool isShift() const {return _code==SHIFT;}
	inline bool isPopRoot() const {return _code==POP_ROOT;}
	inline bool isLeftArc() const {return _code>=LEFT_ARC && _code < RIGHT_ARC;}
	inline bool isRightArc() const {return _code>=RIGHT_ARC;}

public:
	const unsigned long &code() const {return _code;}

	bool operator == (const Actions &a1) const {return _code == a1._code;}
	bool operator != (const Actions &a1) const {return _code != a1._code;}
};
#endif
