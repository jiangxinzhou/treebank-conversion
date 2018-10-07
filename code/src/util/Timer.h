/************************************************************
*
*   Author : Terry Koo
*            maestro@csail.mit.edu
*
*   Timer : utilities for timing runs
*
************************************************************/
#ifndef EGSTRA_TIMER_H
#define EGSTRA_TIMER_H


#include <time.h>
#include <assert.h>
#include <iomanip>

#ifdef _LINUX
#include <sys/time.h>
#endif

/* macro that starts a timer (also allocates local vars) */
#define timer_start()				\
struct timeval tv1;				\
	gettimeofday(&tv1, NULL)

/* macro that stores elapsed seconds in timediff */
#define timer_stop()						\
struct timeval tv2;						\
	gettimeofday(&tv2, NULL);					\
	const double timediff =					\
	(double)tv2.tv_sec   - (double)tv1.tv_sec +			\
	((double)tv2.tv_usec - (double)tv1.tv_usec)/1000000.0;	\
	assert(timediff > 0)

/* macro that prints out a dot every 32 units and a percentage every
256 units.  ii and nn are the current tick and the total ticks */
#define ticker(ii, nn)							\
	if(((ii) & 0x1f) == 0) {						\
	if(((ii) & 0xff) == 0) {					\
	const double pct = 100.0*(double)(ii)/(double)(nn);		\
	cerr << fixed << setprecision(0)				\
	<< "(" << pct << "%)" << flush;				\
	} else {							\
	cerr << "." << flush;						\
	}								\
	}

/* macro that prints out a dot every n1 units and a percentage every
n2 units.  ii and nn are the current tick and the total ticks */
#define ticker4(ii, nn, n1, n2)						\
	if(((ii) & n1) == 0) {						\
	if(((ii) & n2) == 0) {						\
	const double pct = 100.0*(double)(ii)/(double)(nn);		\
	cerr << fixed << setprecision(0)				\
	<< "(" << pct << "%)" << flush;				\
	} else {							\
	cerr << "." << flush;						\
	}								\
	}


namespace egstra {
	/* helper function that pretty-prints a time interval */
	const char* timestr(double seconds);
}


#endif /* EGSTRA_TIMER_H */
