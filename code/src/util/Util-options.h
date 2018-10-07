/**
* Implements management of command line options
* @file Options.h
* @author Mihai Surdeanu
*/

#ifndef EGSTRA_OPTIONS_H
#define EGSTRA_OPTIONS_H

#include <iostream>
#include <string>
#include <string.h>

namespace egstra {

	class options {

	public:
		static void delete_options();

		/** 
		* Reads options from a file 
		*/
		static void read(const std::string & file, bool overwrite = false);

		/** 
		* Writes options to a file 
		*/
		static void write(const std::string & file);

		/** 
		* Reads options from the command line, in the form "--name=value", or
		* in the form "--name", which assumes a default value of 1
		* @return A positive integer representing the first free position in the
		*         argument array. If some error occured, a Boolean exception 
		*         is thrown out of this method
		*/
		static int read(int argc, char ** argv);

		static void display(std::ostream & os, const std::string off="");

		static void set(const std::string & name,
			const std::string & value,
			const bool clobber = false);

		/**
		* Fetches the value of a option
		* If the option is not found in the hash, the environment is inspected
		*/
		static bool get(const std::string & name,
			std::string & value,
			bool useEnvironment = true);

		/** Fetches a double option */
		static bool get(const std::string & name,
			double & value,
			bool useEnvironment = true);

		/** Fetches a int option */
		static bool get(const std::string & name,
			int & value,
			bool useEnvironment = true);

		static bool contains(const std::string & name);


	private:

		static bool readNameValue(const std::string & arg,
			std::string & name,
			std::string & value,
			bool defaultValue = false);

		static bool substitute(const std::string & raw,
			std::string & value);

		/** finds first quote not preceded by backslash */
		static int findQuote(const std::string & input,
			int offset);

		static void* _options;
		static std::string _calledas;
	};

}

#endif /* EGSTRA_OPTIONS_H */
