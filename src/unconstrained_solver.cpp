#include "unconstrained_solver.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

inline std::string vec_to_string(const Eigen::VectorXd& v)
{
    std::ostringstream oss;
    oss << v.transpose();
    return oss.str();
}
namespace furiaoptimizer{

Result Solver::solve(const CostFunc& f, const GradientFunc& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x){

    spdlog::info("Starting solve");

    Result result;
    result.summary.initial_cost = f(params, x);

    int iter = 0;
    double converged = false;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();
    Eigen::VectorXd x_i = x;

    while (iter < options_.max_iter) {

        Eigen::VectorXd g_i = g(params, x_i);
        double f_i = f(params, x_i);

        spdlog::info(
            "iter={},cost={:.8f},grad_norm={:.3e},x={}",
            iter,
            f_i,
            g_i.norm(),
            vec_to_string(x_i)
        );

        auto direction = compute_direction(g_i);

        if ( (g_i.transpose() * direction).norm() <= options_.gradient_tolerance || Dx_i <= options_.step_tolerance || Df_i <= options_.function_tolerance) {
            converged = true;
            break;
        }

        auto step_length = compute_step_length(f, g_i, params, x_i, direction);

        Eigen::VectorXd x_new = x_i + step_length * direction;

        Dx_i = (x_new - x_i).norm()/std::max(x_i.norm(), 1e-16);
        Df_i = std::abs(f(params, x_new) - f_i)/std::max(std::abs(f_i), 1e-16);

        x_i = x_new;
        iter++;
    }

    result.x = x_i;
    result.summary.iterations = iter;
    result.summary.final_cost = f(params, x_i);
    result.summary.converged = converged;
    return result;
};

Eigen::VectorXd Solver::compute_direction(const Eigen::VectorXd& g){
    if (options_.direction_method == DirectionMethod::GradientDescent) {
        return -g;
    }
    else if (options_.direction_method == DirectionMethod::GaussNewton){
        // Placeholder for Gauss-Newton direction computation
        return -g; // This should be replaced with the actual Gauss-Newton direction
    }
    else if (options_.direction_method == DirectionMethod::BFGS){
        // Placeholder for BFGS direction computation
        return -g; // This should be replaced with the actual BFGS direction
    }
    else {
        //Notice that if options_.direction_method == DirectionMethod::ExactNewton is selected the 
        //compute_direction function with the Hessian should be called instead, so we can throw an error here
        throw std::invalid_argument("Unsupported direction method");
    }
}

double Solver::compute_step_length(const CostFunc& f, const Eigen::VectorXd& g, const Eigen::VectorXd& params, const Eigen::VectorXd& x, const Eigen::VectorXd& direction)
{
    if (options_.globalization_method == GlobalizationMethod::LineSearch)
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
    else if (options_.globalization_method == GlobalizationMethod::TrustRegion)
    {
        // Prototype placeholder
        return 1.0;
    }
    else
    {
        throw std::invalid_argument("Unsupported globalization method");
    }
}

}