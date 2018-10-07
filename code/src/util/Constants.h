/**
* Contains all global constants for this project
* @file Constants.h
* @author Mihai Surdeanu
*/

#ifndef EGSTRA_CONSTANTS_H
#define EGSTRA_CONSTANTS_H

#include <climits>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <set>
#include <map>
using namespace std;
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define VERSION "0.2"


/* possible styles of randomization */
#define EGSTRA_RAND_UNIFORM          0
#define EGSTRA_RAND_PERMUTATION      1
#define EGSTRA_DETERMINISTIC         2

/* convert a rand-type to a string description */
#define RAND_LONG_GLOSS(randtype)				\
	(randtype == EGSTRA_RAND_UNIFORM				\
	? "random with replacement"					\
	: (randtype == EGSTRA_RAND_PERMUTATION				\
	? "random permutations"					\
	: (randtype == EGSTRA_DETERMINISTIC				\
	? "deterministic (cyclic)"				\
	: "<unknown randomization, error!>")))			\

/* convert a rand-type to an abbreviation */
#define RAND_SHORT_GLOSS(randtype)				\
	(randtype == EGSTRA_RAND_UNIFORM ? "R"				\
	: (randtype == EGSTRA_RAND_PERMUTATION ? "P"			\
	: (randtype == EGSTRA_DETERMINISTIC ? "D"			\
	: "<error>")))						\


/* multiplicative increment to per-example learning rates in the
adaptive scheme */
#define EGSTRA_LRATE_DEFAULT_BOOST 1.05
/* fraction of examples to sample when automatically estimating the
learning rate, and a minimum sample size as well */
#define EGSTRA_LRATE_SAMPLERATIO   0.1
/* proportion of cuts (relative to number of examples) permissible to
consider a learning rate viable when estimating */
#define EGSTRA_LRATE_CUTRATIO      0.05
/* maximum number of learning-rate halvings per example */
#define EGSTRA_LRATE_MAXCUTS       5
/* convergence tolerance for the relative duality gap (= gap/dual) */
#define EGSTRA_CONV_DUALGAP_TOL  1e-8

/* default DEG regularization path tolerance */
#define EGSTRA_REGPATH_TOL_DEFAULT  1e-3
/* default DEG regularization path step size */
#define EGSTRA_REGPATH_STEP_DEFAULT ((double)sqrt((double)0.5))

/* default beta value */
#define EGSTRA_BETA_DEFAULT 3

/* count cutoff for part-of-speech tagging domains */
#define EGSTRA_TAGDOMAIN_MINFREQ 5

// Maximum line size in a CoNLL-X file
#define MAX_CONLLX_LINE 32 * 1024

// Maximum line size in a feature dictionary file
#define MAX_FEATDICT_LINE 32 * 1024

/* limit to the max number of "frequent" words */
#define EGSTRA_FWLIM_MAX 1000
#define EGSTRA_TAG_FWLIM_MAX 200

// Discard all features with frequency below this threshold
#define UNKNOWN_FEATURE_THRESHOLD 0

// Threshold for the token distance within a dependency
#define TOKEN_DISTANCE_THRESHOLD 41

// Threshold for the number of verbs between the dependency tokens
#define VERB_COUNT_THRESHOLD 4
// Threshold for the number of commas between the dependency tokens
#define PUNC_COUNT_THRESHOLD 4
// Threshold for the number of coordinations between the dependency tokens
#define CC_COUNT_THRESHOLD 4

#endif /* EGSTRA_CONSTANTS_H */



