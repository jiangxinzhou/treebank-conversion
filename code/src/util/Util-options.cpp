/**
 * delete _options;
* Implements management of command line options
* @file Options.cc
* @author Mihai Surdeanu
*/

#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#ifdef _LINUX
#include <unistd.h>
#endif

#include "Util-options.h"
#include "CppAssert.h"
#include "StringMap.h"
#include "CharUtils.h"
#include "MyLib.h"

#define OPTIONS ((StringMap<string>*)_options)

#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

using namespace std;

namespace egstra {

	void* options::_options = NULL;
	string options::_calledas;

	void options::delete_options() {
		delete (StringMap<string>*)_options; 
	}

	void options::set(const string & name,
		const string & value,
		bool clobber) {
			if(clobber) {
				OPTIONS->overwrite(name.c_str(), value);
			} else {
				OPTIONS->set(name.c_str(), value);
			}
	}

	bool options::contains(const string & name) {
		return OPTIONS->contains(name.c_str());
	}


#define MAX_PAR_LINE (16 * 1024)

	/** Returns the position of the first non-space character */
	static int skipSpaces(const char * line)
	{
		int i = 0;
		for(; line[i] != '\0'; i ++){
			if(! isspace(line[i])){
				break;
			}
		}
		return i;
	}

	void options::write(const string& file) {
		FILE* out = fopen(file.c_str(), "w");
		RVASSERT(out != NULL, "couldn't open \"" << file << "\"");
		if(_calledas.size() > 0) {
			fprintf(out, "#!%s\n\n", _calledas.c_str());
		}
		for(StringMap<string>::const_iterator it = OPTIONS->begin();
			it != OPTIONS->end(); ++it) {
				const char* name = it->first;
				const char* value = it->second.c_str();
				fprintf(out, "%s=\"%s\"\n", name, value);
		}
		fclose(out);
	}


	void options::read(const string & file,
		bool overwrite)
	{
		if(_options == NULL) { _options = new StringMap<string>; }

		ifstream is(file.c_str());

		if(! is){
			cerr << "Failed to open file: " << file << endl;
			throw(false);
		}

		char line[MAX_PAR_LINE];
        std::string strLine = "";
		while ( my_getline(is, strLine) ) {
            sprintf(line, "%s", strLine.c_str());
			int start = skipSpaces(line);

			//
			// this is a blank line
			//
			if(line[start] == '\0'){
				continue;
			}

			//
			// this is a comment line
			//
			if(line[start] == '#'){
				continue;
			}

			//
			// read the current option
			//
			string name, value;
			if(readNameValue(line + start, name, value, false) == false){
				cerr << "Failed to parse argument \"" << line + start << "\"" << endl;
				throw(false);
			}

			//
			// add the option to hash
			//
			set(name, value, overwrite); // zhenghua: should like this, not the following
			//if(overwrite == true || contains(name) == false){
			//	set(name, value);
			//}
		}
	}
	

	int options::read(int argc, 
		char ** argv)
	{
		if(_options == NULL) { _options = new StringMap<string>; }

		const char* exec = argv[0];
		if(exec[0] == '/') { /* absolute path to executable */
			_calledas = exec[0];
		} else {
			int size = 256;
			char* tmp = (char*)malloc(size*sizeof(char));
			while(GetCurrentDir(tmp, size) == NULL) {
				size *= 2; tmp = (char*)realloc(tmp, size*sizeof(char));
			}
			string cwd = tmp;
			free(tmp);
			_calledas = cwd + "/" + argv[0];
		}

		int current = 1;

		for(; current < argc; current ++){
			int length = strlen(argv[current]);

			//
			// found something that looks like a option:
			// --name=value
			//
			if(length > 2 && strncmp(argv[current], "--", 2) == 0){
				string arg;
				for(int i = 2; i < length; i ++){
					arg.append(1, (char) argv[current][i]);
				}

				string name, value;
				if(readNameValue(arg, name, value, true) == false){
					cerr << "Failed to parse argument \"" << arg << "\"" << endl;
					throw(false);
				}

				/* clobber anything pre-existing */
				set(name, value, true);
			} else{
				// found something that is not a option
				//	break;
				continue;
			}
		}

		return current;
	}


	bool options::readNameValue(const string & arg, 
		string & name,
		string & value,
		bool defaultValue)
	{
		string sep = "=:";

		int nameStart = arg.find_first_not_of(sep, 0);
		if(nameStart < 0) return false;

		int nameEnd = arg.find_first_of(sep, nameStart);
		if(nameEnd < 0) nameEnd = arg.size();

		name = arg.substr(nameStart, nameEnd - nameStart);

		if(nameEnd < (int) arg.size()){
			int valueStart = arg.find_first_not_of(sep, nameEnd);
			if(valueStart < 0) return false;

			string raw;

			if(arg[valueStart] == '\"'){
				int valueEnd = findQuote(arg, valueStart + 1);
				raw = arg.substr(valueStart + 1, valueEnd - valueStart - 1);
			} else{
				int valueEnd = arg.find_first_of(sep, valueStart);
				if(valueEnd < 0) valueEnd = arg.size();
				raw = arg.substr(valueStart, valueEnd - valueStart);
			}

			if(substitute(raw, value) == false){
				return false;
			}
		} else{
			if(defaultValue == true){
				// Assume a default value of "1"
				value = "1";
			} else{
				return false;
			}
		}

		return true;
	}

	int options::findQuote(const string & input,
		int offset)
	{
		for(int i = offset; i < (int) input.size(); i ++){
			if(input[i] == '\"' &&
				(i == 0 || input[i - 1] != '\\')){
					return i;
			}
		}
		return input.size();
	}

	bool options::substitute(const string & raw,
		string & value)
	{
		ostringstream out;
		int end = -1;

		for(unsigned int i = 0; i < raw.size(); i ++){

			//
			// found a complete variable: ${...}
			//
			if(i < raw.size() - 3 && raw[i] == '$' && raw[i + 1] == '{' &&
				(end = raw.find_first_of('}', i + 2)) > 0){

					string varName = raw.substr(i + 2, end - i - 2);
					string varValue;

					// fetch the variable value
					if(get(varName, varValue, true) == false){
						cerr << "Undefined option: " << varName << endl;
						return false;
					}

					// add this value to stream
					out << varValue;

					i = end;
			} 
			// found a special character preceded by backslash
			else if(raw[i] == '\\' && i < raw.size() - 1){
				out << raw[i + 1];
				i ++;
			} else{
				out << raw[i];
			}
		}

		value = out.str();
		return true;
	}

	bool options::get(const string & name,
		string & value,
		bool useEnvironment)
	{
		if(OPTIONS->get(name.c_str(), value) == true){
			return true;
		}

		if(useEnvironment == false){
			return false;
		}

		// convert the name to regular chars
		string n;
		for(unsigned int i = 0; i < name.size(); i ++){
			n.append(1, (char) name[i]);
		}

		char * env = getenv(n.c_str());

		if(env == NULL){
			return false;
		}

		// convert the value to String
		value.clear();
		int length = strlen(env);
		for(int i = 0; i < length; i ++){
			value.append(1, (char) env[i]);
		}

		return true;
	}

	bool options::get(const string & name,
		double & value,
		bool useEnvironment)
	{
		string str;
		if(get(name, str, useEnvironment) == false){
			return false;
		}

		value = strtod(str.c_str(), NULL);
		return true;
	}

	bool options::get(const string & name,
		int & value,
		bool useEnvironment)
	{
		string str;
		if(get(name, str, useEnvironment) == false){
			return false;
		}

		value = strtol(str.c_str(), NULL, 10);
		return true;
	}

	void options::display(ostream & os, const string off)
	{
		for(StringMap<string>::const_iterator it = OPTIONS->begin();
			it != OPTIONS->end(); it ++){
				os << off << it->first << " = " 
					<< it->second << endl;
		}
	}

	/*
	int main()
	{
	options::read("PARAMS");
	options::display(cerr);
	}
	*/
}
