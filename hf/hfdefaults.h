#ifndef HF_DEFAULTS_H
#define HF_DEFAULTS_H

#include <string>

namespace hf {

static const int HF_PRINT_LEVEL = 0;
static const int HF_MAX_ITER = 100;
static const int HF_DIIS_MAX_VECS = 10;
static const int HF_DIIS_MIN_VECS = 2;
static const int HF_DIIS_START = 0;

static const double HF_SCF_THRESH = 1e-7;
static const double HF_GWH_K = 1.75;
static const double HF_SAD_SCF_THRESH = 1e-4;

static const bool HF_USE_DF = false;
static const bool HF_SCF_DIIS = true;
static const bool HF_DIIS_BETA = true;
static const bool HF_SAD_USE_DF = false;
static const bool HF_SAD_SCF_DIIS = false;
static const bool HF_SAD_SPIN_AVERAGE = true;

static const std::string HF_GUESS = "core";
static const std::string HF_SAD_GUESS = "core";

} // end namesapce

#endif
