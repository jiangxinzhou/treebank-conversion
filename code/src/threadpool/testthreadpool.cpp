/**
* threadpool_test.c, copyright 2001 Steve Gribble
*
* Just a regression test for the threadpool code.
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>
#include "threadpool.h"

#include "spthread.h"

//extern int errno;

void mylog( FILE * fp, const  char  *format,  /*args*/ ...)
{
	va_list ltVaList;
	va_start( ltVaList, format );
	vprintf( format, ltVaList );
	va_end( ltVaList );

	fflush( stdout );
}

void dispatch_threadpool_to_me(void *arg) {
	int *seconds = (int *) arg;

	fprintf(stdout, "  in   dispatch_threadpool %3d %ld\n", *seconds, sp_thread_self());
	sp_sleep(*seconds);
	fprintf(stdout, "  done dispatch_threadpool %3d %ld\n", *seconds, sp_thread_self());
}

int threadpool_main(int argc, char **argv) {
	threadpool tp;

	tp = create_threadpool(2);
    int second = 3;
	fprintf(stdout, "**main** dispatch_threadpool 3\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);
	fprintf(stdout, "**main** dispatch_threadpool 6\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);
	fprintf(stdout, "**main** dispatch_threadpool 7\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);
	fprintf(stdout, "**main** dispatch_threadpool 5\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);
	fprintf(stdout, "**main** dispatch_threadpool 2\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);
	fprintf(stdout, "**main** dispatch_threadpool 4\n");
	dispatch_threadpool(tp, dispatch_threadpool_to_me, (void *) &second);


	fprintf(stdout, "**main done second\n");

	destroy_threadpool( tp );


	return -1;
}

