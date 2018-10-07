#include <iostream>
#include <string>

#include <stdlib.h>

#include "./util/Constants.h"
#include "./util/Util-options.h"

#include "Parser.h"

using namespace std;
using namespace egstra;
using namespace dparser;


int GraphBuilder::max_sentence_length = 0;


int main(int argc, char **argv)
{
	srand(0);
	cerr << argc << endl;
	if (argc < 2) {
		cerr << "cmd format: exe-file config-file-name [options]" << endl;
		exit(-1);
	}
	const char* const arg = argv[1];
	if (arg[0] != '-') {
		options::read(string(argv[1]));
	}
	else {
		cerr << "argv[1] must be the config file" << endl;
		exit(1);
	}

	/* now parse the command-line arguments (potentially overwriting
	anything set in a config file) */
	options::read(argc, argv);
	options::display(cerr, "");

	Parser dparser;
	dparser.run();
	options::delete_options;

	return 0;
}
