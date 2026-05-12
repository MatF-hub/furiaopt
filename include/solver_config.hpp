#pragma once
#include <Eigen/Dense>
#include <optional>
#include "types.hpp"

namespace furiaoptimizer
{

enum class DirectionMethod {
    GradientDescent,
    GaussNewton,
    BFGS,
    ExactNewton
};

enum class GlobalizationMethod {
    LineSearch,
    TrustRegion
};
struct SolverOptions{
    DirectionMethod direction_method;
    GlobalizationMethod globalization_method;
    int max_iter;
    double gradient_tolerance;
    double step_tolerance;
    double function_tolerance;
    std::string log_file_folder_path;
};

struct Problem{
    
    Eigen::VectorXd params;
    Eigen::VectorXd x0;
    CostFunc cost_func;

    //Use it for Newton/Quasi-Newton methods.
    std::optional<GradientFunc> gradient_func;
    std::optional<HessianFunc> hessian_func;

    //Use it for Gauss-Newton methods. The cost function is defined as f(x) = 0.5 * F(x)^T*F(x), where F(x) is a column vector of functions.
    std::optional<ResidualFunc> residual_func;
    std::optional<GradientResidualFunc> gradient_residual_func;
    
    // For Quadratic Programming: f(x) = 0.5 * x'Qx + c'x
    std::optional<Eigen::MatrixXd> Q_quadratic; // Hessian (constant)
    std::optional<Eigen::VectorXd> c_quadratic; // Linear term

    // For Linear Programming: f(x) = c^T*x
    std::optional<Eigen::VectorXd> c_linear; // Linear term

    bool hasGradient() const { return gradient_func.has_value(); }
    bool hasHessian() const { return hessian_func.has_value(); }
    bool isQuadratic() const { return Q_quadratic.has_value() && c_quadratic.has_value(); }
    bool isLinear() const { return !Q_quadratic.has_value() && c_quadratic.has_value(); }
    bool isLeastSquares() const { return residual_func.has_value(); }
};

struct SolverSummary {
    int iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    bool converged = false;
};

struct Result {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(0);
    SolverSummary summary;
};

}