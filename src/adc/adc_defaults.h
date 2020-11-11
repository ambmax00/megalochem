#ifndef ADC_ADC_DEFAULTS_H
#define ADC_ADC_DEFAULTS_H

#include <string>

namespace adc {

static const int ADC_PRINT_LEVEL = 0;
static const int ADC_NROOTS = 1;
static const int ADC_ORDER = 0;
static const int ADC_NBATCHES_X = 5;
static const int ADC_NBATCHES_B = 5;
static const int ADC_NLAP = 5;
static const int ADC_DIAG_ORDER = 0;

static const double ADC_C_OS = 1.3;
static const double ADC_C_OS_COUPLING = 1.17;

static const std::string ADC_METHOD = "ri_adc_1";
static const std::string ADC_ERIS = "core";
static const std::string ADC_METRIC = "coulomb";
static const std::string ADC_BUILD_Z = "llmp_full";
static const std::string ADC_BUILD_J = "dfao";
static const std::string ADC_BUILD_K = "dfao";
static const std::string ADC_INTERMEDS = "core";
static const std::string ADC_DOUBLES = "full";
	
} // end namespace

#endif
