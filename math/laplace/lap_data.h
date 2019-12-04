#ifndef LAP_DATA_H
#define LAP_DATA_H

#include <vector>

namespace math {

struct laplace_data {
        int k;
        double R;
        std::vector<double> omega;
        std::vector<double> alpha;
};

extern std::vector<laplace_data> LAPLACE_DATA;

}

#endif
