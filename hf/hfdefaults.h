#ifndef HF_DEFAULTS_H
#define HF_DEFAULTS_H

#include <string>

namespace hf {

static const int HF_PRINT_LEVEL = 0;
static const int HF_MAX_ITER = 100;

static const double HF_SCF_THRESH = 1e-9;

static const bool HF_USE_DF = false;

static const std::string HF_GUESS = "core";

} // end namesapce

#endif
