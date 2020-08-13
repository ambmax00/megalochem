#ifndef IO_VALID_KEYS_H
#define IO_VALID_KEYS_H
#include "utils/json.hpp"

namespace filio {

using json = nlohmann::json;

static const json valid_keys = 
{
	{"name", "string"},
	{"global", {
		{"block_threshold", 1e-9}, // block threshold for dbcsr
		{"integral_precision", 1-12}, // as the name says
		{"integral_omega", 0.1} // omega factor for erfc integrals
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
		{"ao_split_method", "atomic"}, // atomic or shell
		{"ao_split", 10}
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
		{"build_J", "exact"}, // how Coulomb matrix is constructed
		{"build_K", "exact"}, // how Exchange matrix is constructed
		{"eris", "direct"}, // how eris are held in memory (core/disk/direct)
		{"intermeds", "core"}, // how intermediates are held in memory (core/disk)
		{"df_metric", "coulomb"}, // which metric to use for batchdf
		{"print", 0}, // print level (0, 1 or 2 at the moment, -1 for silent output)
		{"nbatches", 4},
		{"occ_nbatches", 2},
		{"skip", false}, // skip hartree fock and read from files
		{"max_iter", 10},
		{"SAD_guess", "core"},
		{"SAD_diis", true},
		{"SAD_spin_average", true},
		{"dfbasis", "string"},
		{"locc", false},
		{"lvir", false}
	}},
	{"mp", {
		{"print", 0},
		{"nlap", 5}, // number of laplace points
		{"nbatches", 3},
		{"dfbasis", "basis"},
		{"c_os", 1.3},
		{"eris", "core"},
		{"intermeds", "core"},
		{"force_sparsity", false}
	}},
	{"adc", {
		{"print", 0},
		{"c_os", 1.3},
		{"c_os_coupling", 1.15},
		{"dfbasis", "basis"},
		{"nroots", 0},
		{"method", "ADC1"}, /* what method? 
			(ri-adc1, ri-adc2, sos-ri-adc, ao-ri-adc1, ao-ri-adc2) */
		{"diag_order", 0} // at which order to compute the ADC diagonal
	}}	
};

}

#endif
