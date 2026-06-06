#include "solvers/constrained_solver.hpp"
#include "generalization_method.hpp"
#include "compute_gradient.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
namespace furiaoptimizer{

ConstrainedSolver::ConstrainedSolver(const SolverOptions& options, const NLPProblem& problem){
    options_ = options;
    problem_ = problem;

    if (!problem_.hasConstraints()) {
        throw std::invalid_argument("If No constraints are provided, please use the UnconstrainedSolver instead of ConstrainedSolver");
    }

    // Initialize the direction strategy based on the selected method
    switch (options_.direction_method) {
        case DirectionMethod::GradientDescent:
            direction_strategy_ = std::make_unique<GradientDescentDirection>(problem_);
            break;
        case DirectionMethod::BFGS:
            direction_strategy_ = std::make_unique<BFGSDirection>(problem_);
            break;
        case DirectionMethod::ExactNewton:
            spdlog::info("Exact Newton direction selected: notice that for this method to converge the Hessian should be PD or at least SPD");
            direction_strategy_ = std::make_unique<ExactHessianDirection>(problem_);
            break;
        default:
            throw std::invalid_argument("Unsupported direction method");
    }
};

Result ConstrainedSolver::solve(){

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = problem_.cost_func(problem_.x0);

    return result;
};

void ConstrainedSolver::SQP_solver(Result& result, const NLPProblem& problem_){

    int iter = 0;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();
    double lambda_i = 0.0;
    double mu_i = 0.0;
    Eigen::VectorXd x_i = problem_.x0;

    while (iter < options_.max_iter) {
        Eigen::VectorXd grad_f = compute_gradient(problem_, x_i);
        Eigen::MatrixXd grad_eq = problem_.jacobian_equality_constraints_func.value()(x_i);
        Eigen::MatrixXd grad_ineq = problem_.jacobian_inequality_constraints_func.value()(x_i);

        iter++;
    }
};

}