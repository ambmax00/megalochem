#ifndef VALID_KEYS_H
#define VALID_KEYS_H
#include "utils/json.hpp"

using json = nlohmann::json;

static const json valid_keys = 
{
	{"name", "string"},
	{"molecule", {
		{"file", "string"}, // filename
		{"unit", "string"}, // angstrom
		{"basis", "string"}, // basisset name
		{"gen_basis", "basis"}, // custom basis set input
		{"dfbasis", "string"}, // df basis set name
		{"geometry", {0,0,0}}, // mol xyz
		{"symbols", "string"}, // mol elements
		{"mult", 0}, // multiplicity
		{"charge", 0} // total charge
			}},
	{"hf", {
		{"guess", "core"}, // HF guess
		{"scf_thresh", 1e-9}, // convergence criteria
		{"unrestricted", true}, // unrestricted HF calc
		{"diis", true}, // use diis or not
		{"diis_max_vecs", 8}, // maximum number of diis vectors in subsdpace
		{"diis_min_vecs", 2}, // minimum number of diis vectors in subspace
		{"diis_start", 0}, // at what iteration to start diis
		{"diis_beta", true}, // whether to use separate coeficients for beta
		{"use_df", false}, // whether to use df 
		{"print", 0} // print level (0, 1 or 2 at the moment, -1 for silent output)
	}}
};

#endif
