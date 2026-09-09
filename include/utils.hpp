#pragma once

#include <iostream> 
#include <Eigen/Dense>

namespace furiaopt::utils {

    inline std::string vec_to_string(const Eigen::VectorXd& v)
    {
        std::ostringstream oss;
        oss.precision(17);  // full double precision -- default is lossy
        oss << v.transpose();
        return oss.str();
    }
}