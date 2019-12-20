#ifndef VALID_KEYS_H
#define VALID_KEYS_H
#include "utils/json.hpp"

using json = nlohmann::json;

static const json valid_keys = 
{
	{"name", "string"},
	{"molecule", {
		{"file", "string"},
		{"unit", "string"},
		{"basis", "string"},
		{"geometry", {0,0,0}},
		{"symbols", "string"},
		{"mult", 0},
		{"charge", 0}
	}}
};

#endif
