/**
* threadpool.c
*
* This file will contain your implementation of a threadpool.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "threadpool.h"
#include "spthread.h"


/* lzh: the following is critical to understand (but maybe a little confusing).
	struct _thread is a data structure to manage a real thread: 
		create a new thread, destroy the thread, assign a new job to the thread when it is idle.

	When creating a new thread, two things need to be done:
		1. allocate struct _thread
			_thread * thread = ( _thread * )malloc( sizeof( _thread ) );
		2. sp_thread_create( &thread->id, &attr, wrapper_fn, (void*)_thread )
		After creation, the thread will keep runing jobs (including waiting for new jobs when being idle)

*/

typedef struct _thread_st {
	sp_thread_t id;
	threadpool parent;

	// lzh: wait until a new job is assinged to this thread.
	sp_thread_mutex_t mutex;
	sp_thread_cond_t cond; 
	
	// lzh: the specific job to do
	dispatch_fn fn;
	void *arg;
} _thread;


// _threadpool is the internal threadpool structure that is
// cast to type "threadpool" before it given out to callers
typedef struct _threadpool_st {
	// you should fill in this structure with whatever you need

	sp_thread_mutex_t tp_mutex; // lzh: control the pool
	sp_thread_cond_t tp_idle; // lzh: there is an idle thread
	sp_thread_cond_t tp_full; // lzh: all threads are idle
	sp_thread_cond_t tp_empty; // lzh: no thread is created

	/* 
		lzh: When a _thread completes its job (idle), it is stored into tp_list
		When a _thread in tp_list is needed to do a new job, the _thread is removed and the position is cleared (=NULL)
	*/
	_thread ** tp_list; // lzh: ( _thread ** )malloc( sizeof( void * ) * MAXT_IN_POOL );

	int tp_index;	// lzh: the number of idle threads in tp_list

	int tp_max_index;  // lzh: num_threads_in_pool (const int)

	int tp_stop; //lzh:  whether all the threads in the pool should stop running (end itself from infinite loop)? tp_stop=1 only when destroy_threadpool()

	int tp_total; // lzh: the number of created threads, maybe less than tp_max_index

} _threadpool;


threadpool create_threadpool(int num_threads_in_pool)
{
	_threadpool *pool;

	// sanity check the argument
	if ((num_threads_in_pool <= 0) || (num_threads_in_pool > MAXT_IN_POOL)) {
		fprintf(stderr, "num_threads_in_pool error: (0, %d]\n", MAXT_IN_POOL);
		return NULL;
	}

	pool = (_threadpool *) malloc(sizeof(_threadpool));
	if (pool == NULL) {
		fprintf(stderr, "Out of memory creating a new threadpool!\n");
		return NULL;
	}

	// add your code here to initialize the newly created threadpool
	sp_thread_mutex_init( &pool->tp_mutex, NULL );
	sp_thread_cond_init( &pool->tp_idle, NULL );
	sp_thread_cond_init( &pool->tp_full, NULL );
	sp_thread_cond_init( &pool->tp_empty, NULL );
	pool->tp_max_index = num_threads_in_pool;
	pool->tp_index = 0;
	pool->tp_stop = 0;
	pool->tp_total = 0;
	pool->tp_list = ( _thread ** )malloc( sizeof( void * ) * MAXT_IN_POOL );
	memset( pool->tp_list, 0, sizeof( void * ) * MAXT_IN_POOL );

	return (threadpool) pool;
}

// lzh: save the created _thread into tp_list of the pool
int save_thread( _threadpool * pool, _thread * thread )
{
	int ret = -1;

	sp_thread_mutex_lock( &pool->tp_mutex );

	if( pool->tp_index < pool->tp_max_index ) {
		pool->tp_list[ pool->tp_index ] = thread;
		pool->tp_index++;
		ret = 0;

		sp_thread_cond_signal( &pool->tp_idle );

		if( pool->tp_index >= pool->tp_total ) {
			sp_thread_cond_signal( &pool->tp_full );
		}
	}

	sp_thread_mutex_unlock( &pool->tp_mutex );

	return ret;
}

// lzh: this is where the job is really done
sp_thread_result_t SP_THREAD_CALL wrapper_fn( void * arg )
{
	_thread * thread = (_thread*)arg;
	_threadpool * pool = (_threadpool*)thread->parent;

	// lzh: this loop is ended when the pool is destroyed or the thread can not be stored into the pool
	for( ; 0 == ((_threadpool*)thread->parent)->tp_stop; ) {
		thread->fn( thread->arg ); // lzh: run the job

		if( 0 != ((_threadpool*)thread->parent)->tp_stop ) break;

		sp_thread_mutex_lock( &thread->mutex ); 

		// lzh: after the job is done, the thread is stored into tp_list
		if( 0 == save_thread( (_threadpool*)thread->parent, thread ) ) {
			sp_thread_cond_wait( &thread->cond, &thread->mutex ); // lzh: the thread will wait until it is activated.
			sp_thread_mutex_unlock( &thread->mutex ); 
			// lzh: before the _thread does a new job, the mutex must be unlocked to avoid deadlock. 
			// lzh: then, do the new job (through the loop control)
		} else {
			sp_thread_mutex_unlock( &thread->mutex );
			sp_thread_cond_destroy( &thread->cond );
			sp_thread_mutex_destroy( &thread->mutex );

			free( thread );
			break;
		}
	}

	sp_thread_mutex_lock( &pool->tp_mutex );
	pool->tp_total--; // decrease the number of created threads
	if( pool->tp_total <= 0 ) sp_thread_cond_signal( &pool->tp_empty );
	sp_thread_mutex_unlock( &pool->tp_mutex );

	return 0;
}

/* lzh: assign a thread to do the job
        create a new thread
        or
        wait for and use an idle thread
*/

int dispatch_threadpool(threadpool from_me, dispatch_fn dispatch_to_here, void *arg)
{
	int ret = 0;

	_threadpool *pool = (_threadpool *) from_me;
	sp_thread_attr_t attr;
	_thread * thread = NULL;

	// add your code here to dispatch a thread
	sp_thread_mutex_lock( &pool->tp_mutex );

	// lzh: the number of created threads reaches maximum, but no one is idle. 
	while( pool->tp_index <= 0 && pool->tp_total >= pool->tp_max_index ) {
		sp_thread_cond_wait( &pool->tp_idle, &pool->tp_mutex );
	}

	// lzh: the number of created threads is less than the maximum. 
	// lzh: create a new thread
	if( pool->tp_index <= 0 ) { 
		_thread * thread = ( _thread * )malloc( sizeof( _thread ) );
		memset( &( thread->id ), 0, sizeof( thread->id ) );
		sp_thread_mutex_init( &thread->mutex, NULL );
		sp_thread_cond_init( &thread->cond, NULL );
		thread->fn = dispatch_to_here;
		thread->arg = arg;
		thread->parent = pool;

		sp_thread_attr_init( &attr );
		sp_thread_attr_setdetachstate( &attr, SP_THREAD_CREATE_DETACHED );

		if( 0 == sp_thread_create( &thread->id, &attr, wrapper_fn, thread ) ) { // create a new thread
			pool->tp_total++;
			//printf( "create thread#%ld\n", thread->id );
		} else {
			ret = -1;
			printf( "\n\n-------- cannot create thread -------- \n" );
			printf( "\n\n-------- what to do?? -------- \n" );
			sp_thread_mutex_destroy( &thread->mutex );
			sp_thread_cond_destroy( &thread->cond );
			free( thread );
		}
	} else {
		pool->tp_index--; 
		thread = pool->tp_list[ pool->tp_index ]; // lzh: use an idle thread.
		pool->tp_list[ pool->tp_index ] = NULL; // lzh: once a thread is occupied, it is removed from tp_list

		thread->fn = dispatch_to_here;
		thread->arg = arg;
		thread->parent = pool;

		sp_thread_mutex_lock( &thread->mutex );
		sp_thread_cond_signal( &thread->cond ) ; // lzh: activate this thread
		sp_thread_mutex_unlock ( &thread->mutex );
	}

	sp_thread_mutex_unlock( &pool->tp_mutex );

	return ret;
}

void wait_all_jobs_done(threadpool me)
{
	_threadpool *pool = (_threadpool *) me;

	sp_thread_mutex_lock( &pool->tp_mutex );

	while ( pool->tp_index < pool->tp_total ) { 
	// if ( pool->tp_index < pool->tp_total ) {  // lzh: some incomplete jobs may be left, must use "while", I do not know why!!
		//printf( "waiting for %d thread(s) to finish\n", pool->tp_total - pool->tp_index );
		sp_thread_cond_wait( &pool->tp_full, &pool->tp_mutex );
	}

	sp_thread_mutex_unlock( &pool->tp_mutex );
}


void destroy_threadpool(threadpool destroyme)
{
	_threadpool *pool = (_threadpool *) destroyme;

	// add your code here to kill a threadpool
	int i = 0;

	sp_thread_mutex_lock( &pool->tp_mutex );

	while ( pool->tp_index < pool->tp_total ) { 
	// if ( pool->tp_index < pool->tp_total ) {  // lzh: some incomplete jobs may be left, must use "while", I do not know why!!
		printf( "waiting for %d thread(s) to finish\n", pool->tp_total - pool->tp_index );
		sp_thread_cond_wait( &pool->tp_full, &pool->tp_mutex );
	}

	pool->tp_stop = 1; // lzh: let all the threads to kill themselves

	for( i = 0; i < pool->tp_index; i++ ) {
		_thread * thread = pool->tp_list[ i ];

		sp_thread_mutex_lock( &thread->mutex );
		sp_thread_cond_signal( &thread->cond ) ; // lzh: activate the idle threads so that they can kill themsevles
		sp_thread_mutex_unlock ( &thread->mutex );
	}

	if( pool->tp_total > 0 ) {
		printf( "waiting for %d thread(s) to exit\n", pool->tp_total );
		sp_thread_cond_wait( &pool->tp_empty, &pool->tp_mutex );
	}

	for( i = 0; i < pool->tp_index; i++ ) {
		free( pool->tp_list[ i ] );
		pool->tp_list[ i ] = NULL;
	}

	sp_thread_mutex_unlock( &pool->tp_mutex );

	pool->tp_index = 0;

	sp_thread_mutex_destroy( &pool->tp_mutex );
	sp_thread_cond_destroy( &pool->tp_idle );
	sp_thread_cond_destroy( &pool->tp_full );
	sp_thread_cond_destroy( &pool->tp_empty );

	free( pool->tp_list );
	free( pool );
}

