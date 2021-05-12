#ifndef UTIL_ELE_TO_INT_H
#define UTIL_ELE_TO_INT_H

#include <map>
#include <string>

namespace util {

static std::map<std::string,int> ele_to_int =
{
	{"X",0},
	{"H",1},
	{"He",2},
	{"Li",3},
	{"Be",4},
	{"B",5},
	{"C",6},
	{"N",7},
	{"O",8},
	{"F",9},
	{"Ne",10}
};

static std::vector<std::string> int_to_ele = {
	"X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"
};

}

#endif
