#ifndef UTIL_ELE_TO_INT_H
#define UTIL_ELE_TO_INT_H

#include <map>
#include <string>

namespace util {

// clang-format off
static inline std::map<std::string, int> ele_to_int = {
    {"X",0},
    {"H",1}, 
    {"He",2}, 
    {"Li",3}, 
    {"Be", 4}, 
    {"B", 5}, 
    {"C", 6}, 
    {"N", 7}, 
    {"O", 8}, 
    {"F", 9}, 
    {"Ne", 10},
    {"Na", 11}, 
    {"Mg", 12}, 
    {"Al", 13}, 
    {"Si", 14}, 
    {"P", 15}, 
    {"S", 16}, 
    {"Cl", 17}, 
    {"Ar", 18}
};

static inline std::vector<std::string> int_to_ele = {
  "X", 
  "H", 
  "He", 
  "Li", 
  "Be", 
  "B",
  "C", 
  "N", 
  "O",  
  "F",  
  "Ne",
  "Na",
  "Mg",
  "Al",
  "Si",
  "P",
  "S",
  "Cl",
  "Ar"
};
// clang-format on

}  // namespace util

#endif
