#include "solvers/qp_solver.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
namespace furiaoptimizer{

QPSolver::QPSolver(const SolverOptions& options, const QPProblem& problem) : options_(std::cref(options)), problem_(std::cref(problem)) {

    cost_func_ = [&problem](const Eigen::VectorXd& x) {
        return 0.5 * x.transpose() * problem.H * x + problem.c.dot(x);
    };
};

Result QPSolver::solve(){

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = cost_func_(problem_.get().x0);

    if (problem_.get().hasInequalityConstraints())
    {
        general_QP_solver(result);
    }
    else if (problem_.get().hasEqualityConstraints())
    {
        equality_constrained_QP_solver(result);
    }
    else
    {
        no_constraints_QP_solver(result);
    }
    return result;
};


void QPSolver::no_constraints_QP_solver(Result& result)
{
    // For quadratic problems w/o constraints, we can directly compute the optimal solution by solving Hx + c = 0
    Eigen::MatrixXd H = problem_.get().H;
    Eigen::VectorXd c = problem_.get().c;
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    if (llt.info() == Eigen::Success) {
        result.x = llt.solve(-c);
    } else {
        // If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
        Eigen::LDLT<Eigen::MatrixXd> ldlt(H);
        if (ldlt.info() == Eigen::Success) {
            result.x = ldlt.solve(-c);
        } else {
            throw std::runtime_error("Failed to solve quadratic problem: Hessian is not semi-positive definite");
        }
    }
    result.summary.final_cost = cost_func_(result.x);
    result.summary.iterations = 0;
    result.summary.converged = true;
};

}