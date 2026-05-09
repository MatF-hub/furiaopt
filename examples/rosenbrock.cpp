#include <Eigen/Dense>
#include "unconstrained_solver.hpp"
#include <iostream>

// The Rosenbrock function itself
double rosenbrock(const Eigen::VectorXd& params, const Eigen::VectorXd& x){
    double A = params[0];
    double B = params[1];
    return std::pow(A - x[1], 2) + B * std::pow(x[2] - std::pow(x[2], 2), 2);
}

int main()
{
    furiaoptimizer::Solver default_solver;

    Eigen::VectorXd x_init(2);
    x_init << 10, -10;

    Eigen::VectorXd params(2);
    // Constants for the Rosenbrock function A = 1.0, B = 100.0
    params << 1.0, 100.0;

    default_solver.Solve(rosenbrock, params, x_init);

    std::cout << "Optimized parameters: " << x_init.transpose() << std::endl;
}