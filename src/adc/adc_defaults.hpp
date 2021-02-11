#ifndef ADC_ADC_DEFAULTS_H
#define ADC_ADC_DEFAULTS_H

#include <string>

namespace adc {

// global 
inline const int ADC_PRINT_LEVEL = 0;
inline const int ADC_NROOTS = 1;
inline const int ADC_NBATCHES_X = 5;
inline const int ADC_NBATCHES_B = 5;
inline const int ADC_NGUESSES = 1;

inline const bool ADC_DO_ADC2 = false;
inline const bool ADC_BLOCK = false;
inline const bool ADC_BALANCING = false;

// ADC1
inline const std::string ADC_ADC1_DF_METRIC = "coulomb";
inline const std::string ADC_ADC1_JMETHOD = "dfao";
inline const std::string ADC_ADC1_KMETHOD = "dfao";
inline const std::string ADC_ADC1_ERIS = "core";
inline const std::string ADC_ADC1_INTERMEDS = "core";

inline const double ADC_ADC1_DAV_CONV = 1e-5;

inline const int ADC_ADC1_MAXITER = 50;

// ADC2
inline const int ADC_ADC2_NLAP = 5;
inline const int ADC_ADC2_MICRO_MAXITER = 50;
inline const int ADC_ADC2_MACRO_MAXITER = 30;

inline const bool ADC_ADC2_LOCAL = false;

inline const std::string ADC_ADC2_DF_METRIC = "coulomb";
inline const std::string ADC_ADC2_JMETHOD = "dfao";
inline const std::string ADC_ADC2_KMETHOD = "dfao";
inline const std::string ADC_ADC2_ZMETHOD = "llmpfull";
inline const std::string ADC_ADC2_ERIS = "core";
inline const std::string ADC_ADC2_INTERMEDS = "core";

inline const double ADC_ADC2_C_OS = 1.3;
inline const double ADC_ADC2_C_OS_COUPLING = 1.17;
inline const double ADC_ADC2_DAV_CONV = 1e-5;
inline const double ADC_ADC2_MICRO_CONV = 1e-6;
inline const double ADC_ADC2_MACRO_CONV = 1e-6;

} // end namespace

#endif
