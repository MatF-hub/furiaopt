#pragma once
#include <Eigen/Dense>

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