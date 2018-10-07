
Zhenghua Li
zhenghualiir@gmail.com
2012.07.02

This package provides useful and simple interfaces to create, control, and destroy a thread pool.
Users do not need to worry about the details about the complex thread management.


I have modified the original package and added comments according to my understanding.
The original version: http://code.google.com/p/spserver/downloads/detail?name=threadpool-0.2.1.src.tar.gz&can=2&q=

1. A bug is fixed: run-time error @Windwos. 
		destroy_threadpool():
				// if ( pool->tp_index < pool->tp_total ) {		
				=> 
				while ( pool->tp_index < pool->tp_total ) {
				
				

2. add a function: void wait_all_jobs_done(threadpool me)..


========================

How to use this package? A real example:


class Parser	{
	
	struct parse_thread_arg {
			parse_thread_arg(Parser * const parser, Instance * const inst) 
				: _parser(parser), _inst(inst) {}

			Parser * const _parser;
			Instance * const _inst;
	};
	
	
	private:
		int _thread_num;
		threadpool _tp;
		
	public:
		Parser() {
				_tp = create_threadpool(_thread_num);
		}

		~Parser(void) {
				destroy_threadpool(_tp);
		}
		
		void parse(Decoder *decoder, Instance *inst) {
				// the codes for parsing the sentence (instance)
		}
		
		void parallel_parse_a_corpus( IOPipe &pipe )
		{
				const int inst_num = pipe.getInstanceNum();		// The sentences in the corpus have been read by IOPipe previously.
				
				// dispatch all the jobs, let the thread-pool parse all sentences
				for (int i = 0; i < inst_num ; ++i)
				{
					Instance *inst = pipe.getInstance(i);
					parse_thread_arg * arg = new parse_thread_arg(this, inst);
					dispatch_threadpool(_tp, parse_thread, (void *)arg);
				}
	
				wait_all_jobs_done(_tp);
				cerr << "\ninstance num: " << inst_num; print_time();
				
				// do other stuff
				// ...
		}
		
		
		static void parse_thread(void *arg) {  // "static" is important!
				parse_thread_arg * _arg = (parse_thread_arg *) arg;
				cerr << "\n\tstart processing inst " << _arg->_inst->id << endl;
				Decoder *decoder = new_decoder();
				_arg->_parser->parse(decoder, _arg->_inst);
				delete_decoder(decoder);
				cerr << "\n\tend   processing inst " << _arg->_inst->id << endl;
				delete _arg;
		}		
}
		
		
	


