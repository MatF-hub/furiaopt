#pragma once
#include "types.hpp"
#include "solver_config.hpp"

#include <Eigen/Dense>

namespace furiaoptimizer {

double compute_step_length(const SolverOptions& options, const CostFunc& f, const Eigen::VectorXd& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& direction)
{
    if (options.globalization_method == GlobalizationMethod::LineSearch)
    {
        // -----------------------------------------
        // Backtracking Line Search (Armijo)
        // -----------------------------------------
        const double beta   = 0.5;     // shrink factor (0,1)
        const double c1     = 1e-4;    // Armijo constant
        const int max_ls    = 25;

        double alpha = 1.0;

        // Current cost
        const double fx = f(params, x);

        // Directional derivative
        const double slope0 = g.dot(direction);

        // Safety check: direction should be descent
        if (slope0 >= 0.0)
        {
            return 0.0;
        }

        for (int i = 0; i < max_ls; ++i)
        {
            Eigen::VectorXd x_trial = x + alpha * direction;

            double f_trial = f(params, x_trial);

            // Armijo condition:
            // f(x + alpha p) <= f(x) + c1 alpha g^T p
            if (f_trial <= fx + c1 * alpha * slope0)
            {
                return alpha;
            }

            alpha *= beta;
        }

        // fallback if line search fails
        return alpha;
    }
    else if (options.globalization_method == GlobalizationMethod::TrustRegion)
    {
        // Prototype placeholder
        return 1.0;
    }
    else
    {
        throw std::invalid_argument("Unsupported globalization method");
    }
};

}