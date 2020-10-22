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
		{"integral_omega", 0.1}, // omega factor for erfc integrals
		{"_required", {"none"}}
	}},
	{"molecule", {
		{"file", "string"}, 
		{"name", "string"}, // filename.xyz
		{"reorder", true}, // whether to reorder atoms
		{"unit", "string"}, // angstrom
		{"basis", "string"}, // basisset name
		{"augmentation", "string"}, // use augmented functions from aug-{basname}
		//{"gen_basis", "basis"}, // custom basis set input
		{"geometry", {0.0,0.0,0.0}}, // mol xyz
		{"symbols", "string"}, // mol elements
		{"mult", 0u}, //multiplicity
		{"charge", 0}, // total charge
		{"mo_split", 10u},
		{"ao_split_method", "atomic"}, // atomic or shell
		{"ao_split", 10u},
		{"_required", {"name", "basis", "mult", "charge"}}
	}},
	{"hf", {
		{"guess", "core"}, // HF guess
		{"scf_thresh", 1e-9}, // convergence criteria
		{"unrestricted", true}, // unrestricted HF calc
		{"diis", true}, // use diis or not
		{"diis_max_vecs", 8u}, // maximum number of diis vectors in subsdpace
		{"diis_min_vecs", 2u}, // minimum number of diis vectors in subspace
		{"diis_start", 0u}, // at what iteration to start diis
		{"diis_beta", true}, // whether to use separate coeficients for beta
		{"build_J", "exact"}, // how Coulomb matrix is constructed
		{"build_K", "exact"}, // how Exchange matrix is constructed
		{"eris", "direct"}, // how eris are held in memory (core/disk/direct)
		{"intermeds", "core"}, // how intermediates are held in memory (core/disk)
		{"df_metric", "coulomb"}, // which metric to use for batchdf
		{"print", 0u}, // print level (0, 1 or 2 at the moment, -1 for silent output)
		{"nbatches_x", 4u},
		{"nbatches_b", 4u},
		{"occ_nbatches", 2u},
		{"skip", false}, // skip hartree fock and read from files
		{"max_iter", 10u},
		{"SAD_guess", "core"},
		{"SAD_diis", true},
		{"SAD_spin_average", true},
		{"dfbasis", "string"},
		{"df_augmentation", true},
		{"locc", false},
		{"lvir", false},
		{"_required", {"dfbasis"}}
	}},
	{"mp", {
		{"print", 0u},
		{"nlap", 5u}, // number of laplace points
		{"nbatches_b", 3u},
		{"nbatches_x", 3u},
		{"dfbasis", "basis"},
		{"df_augmentation", true},
		{"c_os", 1.3},
		{"eris", "core"},
		{"intermeds", "core"},
		//{"force_sparsity", false},
		{"build_Z", "LLMPFULL"},
		{"_required", {"dfbasis"}}
	}},
	{"adc", {
		{"print", 1u},
		{"c_os", 1.3},
		{"c_os_coupling", 1.15},
		{"dfbasis", "basis"},
		{"df_augmentation", true},
		{"nbatches_b", 3u},
		{"nbatches_x", 3u},
		{"nlap", 5u},
		{"nroots", 1u},
		{"method", "ADC1"}, /* what method? 
			(ri-adc1, ri-adc2, sos-ri-adc, ao-ri-adc1, ao-ri-adc2) */
		{"diag_order", 0}, // at which order to compute the ADC diagonal
		{"build_J", "df"},
		{"build_K", "df"},
		{"build_Z", "LLMPFULL"},
		{"eris", "core"},
		{"intermeds", "core"},
		{"doubles", "full"},
		{"_required", {"nroots", "dfbasis", "method"}}
	}},
	{"_required", {"molecule", "hf"}}	
};

}

#endif
