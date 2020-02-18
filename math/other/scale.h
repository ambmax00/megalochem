#ifndef MATH_SCALE_H
#define MATH_SCALE_H

#include "math/tensor/dbcsr.hpp"
#include "utils/params.hpp"
#include <vector>

namespace math {

// do t[ij] = t[ij] * v[j]
struct scale_params {
	required<dbcsr::tensor<2>,ref> t_in;
	required<vec<double>,ref> v_in;
	optional<vec<int>,val> bounds;
};
void scale(scale_params&& p);

}

#endif
