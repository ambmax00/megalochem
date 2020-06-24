#ifndef VALID_KEYS_H
#define VALID_KEYS_H
#include "utils/json.hpp"

using json = nlohmann::json;

static const json valid_keys = 
{
	{"name", "string"},
	{"global", {
		{"batchsize", 1000}, // batch size for tensors, in megabytes
		{"block_threshold", 1e-9} // block threshold for dbcsr
	}},
	{"molecule", {
		{"file", "string"}, 
		{"name", "string"}, // filename.xyz
		{"reorder", true}, // whether to reorder atoms
		{"unit", "string"}, // angstrom
		{"basis", "string"}, // basisset name
		{"gen_basis", "basis"}, // custom basis set input
		{"geometry", {0,0,0}}, // mol xyz
		{"symbols", "string"}, // mol elements
		{"mult", 0}, // multiplicity
		{"charge", 0}, // total charge
		{"mo_split", 10},
		{"ao_split_method", "atomic"} // atomic or shell
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
		{"build_J", "exact"},
		{"build_K", "exact"},
		{"print", 0}, // print level (0, 1 or 2 at the moment, -1 for silent output)
		{"skip", false}, // skip hartree fock and read from files
		{"max_iter", 10},
		{"SAD_guess", "core"},
		{"SAD_diis", true},
		{"SAD_use_df", true},
		{"SAD_spin_average", true},
		{"dfbasis", "string"} 
	}},
	{"mp", {
		{"print", 0},
		{"nlap", 5}, // number of laplace points
		{"dfbasis", "basis"},
		{"c_os", 1.3}
	}},
	{"adc", {
		{"print", 0},
		{"c_os", 1.3},
		{"c_os_coupling", 1.15},
		{"nroots", 0},
		{"order", 0}, // wether to do ADC(0), ADC(1), ...
		{"use_ao", false}, // use AO basis formulation
		{"use_sos", false}, // use SOS approximation
		{"use_lp", false}, // use laplace transform
		{"diag_order", 0} // at which order to compute the ADC diagonal
	}}	
};

#endif
