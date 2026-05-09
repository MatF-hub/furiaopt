#include <Eigen/Dense>
#include "unconstrained_solver.hpp"
#include <iostream>

// The Rosenbrock function itself
#include <Eigen/Dense>
#include <cmath>

double rosenbrock(const Eigen::VectorXd& params, const Eigen::VectorXd& x) {
    double a = params[0];
    double b = params[1];
    return std::pow(a - x[0], 2) + b * std::pow(x[1] - std::pow(x[0], 2), 2);
}

Eigen::VectorXd rosenbrock_gradient(const Eigen::VectorXd& params, const Eigen::VectorXd& x) {
    double a = params[0];
    double b = params[1];
    Eigen::VectorXd grad(2);
    // Derivative with respect to x[0]
    grad[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - std::pow(x[0], 2));
    // Derivative with respect to x[1]
    grad[1] = 2 * b * (x[1] - std::pow(x[0], 2));
    return grad;
}

int main()
{
    furiaoptimizer::Solver default_solver;

    Eigen::VectorXd x_init(2);
    x_init << 10, -10;

    Eigen::VectorXd params(2);
    // Constants for the Rosenbrock function A = 1.0, B = 100.0
    params << 1.0, 100.0;

    furiaoptimizer::Result result = default_solver.solve(rosenbrock, rosenbrock_gradient, params, x_init);

    std::cout << "Optimized parameters: " << result.x.transpose() << std::endl;
    std::cout << "Solver summary: " << std::endl;
    std::cout << "  Iterations: " << result.summary.iterations << std::endl;
    std::cout << "  Initial cost: " << result.summary.initial_cost << std::endl;
    std::cout << "  Final cost: " << result.summary.final_cost << std::endl;
    std::cout << "  Converged: " << (result.summary.converged ? "Yes" : "No") << std::endl;
}