#include "unconstrained_solver.hpp"
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

Solver::Solver(const SolverOptions& options, const Problem& problem){
    options_ = options;
    problem_ = problem;

    // Initialize the direction strategy based on the selected method
    switch (options_.direction_method) {
        case DirectionMethod::GradientDescent:
            direction_strategy_ = std::make_unique<GradientDescentDirection>(problem_);
            break;
        case DirectionMethod::GaussNewton:
            spdlog::info("Gauss-Newton direction selected: notice that for this method to work the cost function structure should be formulated in a least squares form. i.e f(x) = 0.5*F(x)^T*F(x) where F is a column vector of residuals");
            direction_strategy_ = std::make_unique<GaussNewtonDirection>(problem_);
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

Result Solver::solve(){

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = problem_.cost_func(problem_.params, problem_.x0);

    if (problem_.isQuadratic())
    {
        spdlog::info("Quadratic problem detected, using direct solver");
        // For quadratic problems, we can directly compute the optimal solution by solving Qx + c = 0
        Eigen::MatrixXd Q = problem_.Q_quadratic.value();
        Eigen::VectorXd c = problem_.c_quadratic.value();
        Eigen::LLT<Eigen::MatrixXd> llt(Q);
        if (llt.info() == Eigen::Success) {
            result.x = llt.solve(-c);
        } else {
            // If LLT fails, try LDLT decomposition, can handle both positive and negative semi definite Hessian
            Eigen::LDLT<Eigen::MatrixXd> ldlt(Q);
            if (ldlt.info() == Eigen::Success) {
                result.x = ldlt.solve(-c);
            } else {
                throw std::runtime_error("Failed to solve quadratic problem: Hessian is not semi-positive definite");
            }
        }
        result.summary.final_cost = problem_.cost_func(problem_.params, result.x);
        result.summary.iterations = 0;
        result.summary.converged = true;
    }
    else if (problem_.isLinear())
    {
        spdlog::info("Linear problem detected w/o constraints detected, checking for unboundedness");
        // For linear problems, we can directly compute the optimal solution by solving c = 0
        Eigen::VectorXd c = problem_.c_linear.value();
        if (c.norm() > 1e-16) {
            throw std::runtime_error("Unbounded cost function detected: for unconstrained optimization the cost function should be at least quadratic to have a well-defined minimum");
        }
    }
    else
    {
        spdlog::info("Non-linear problem detected, using iterative solver");
        non_linear_solver(result, problem_);
    }

    return result;
};


void Solver::non_linear_solver(Result& result, const Problem& problem_){

    int iter = 0;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();
    Eigen::VectorXd x_i = problem_.x0;

    while (iter < options_.max_iter) {

        Eigen::VectorXd g_i = compute_gradient(problem_, x_i);

        double f_i = problem_.cost_func(problem_.params, x_i);

        spdlog::info(
            "iter={},cost={:.8f},grad_norm={:.3e},x={}",
            iter,
            f_i,
            g_i.norm(),
            vec_to_string(x_i)
        );

        auto direction = direction_strategy_->getDirection(g_i, x_i);

        if ((g_i.transpose() * direction).norm() <= options_.gradient_tolerance || Dx_i <= options_.step_tolerance || Df_i <= options_.function_tolerance) {
            break;
        }

        auto step_length = compute_step_length(options_, problem_.cost_func, g_i, problem_.params, x_i, direction);

        Eigen::VectorXd x_new = x_i + step_length * direction;

        Dx_i = (x_new - x_i).norm()/std::max(x_i.norm(), 1e-16);
        Df_i = std::abs(problem_.cost_func(problem_.params, x_new) - f_i)/std::max(std::abs(f_i), 1e-16);

        x_i = x_new;
        iter++;
    }

    result.x = x_i;
    result.summary.iterations = iter;
    result.summary.final_cost = problem_.cost_func(problem_.params, x_i);
    result.summary.converged = iter < options_.max_iter;
};

}