#include "solvers/lp_solver.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
namespace furiaoptimizer{

LPSolver::LPSolver(const SolverOptions& options, const LPProblem& problem) : options_(std::cref(options)), problem_(std::cref(problem)) {
    cost_func_ = [&problem](const Eigen::VectorXd& x) {
        return problem.c.transpose() * x;
    };
};

Result LPSolver::solve(){

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = cost_func_(problem_.get().x0);

    if (problem_.get().hasInequalityConstraints())
    {
        general_LP_solver(result);
    }
    else if (problem_.get().hasEqualityConstraints())
    {
        equality_constrained_LP_solver(result);
    }
    else
    {
        spdlog::error("No constraints LP solver does not make sense");
        throw std::runtime_error("No constraints LP solver does not make sense");
    }

    return result;
};



}