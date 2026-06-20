#include "solvers/constrained_solver.hpp"
#include "solvers/qp_solver.hpp"
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
namespace furiaopt{

ConstrainedSolver::ConstrainedSolver(const ConstrainedSolverOptions& options, const NLPProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0){

    if (!problem.hasEqualityConstraints() && !problem.hasInequalityConstraints()) {
        spdlog::warn("No constraints provided for the problem, call unconstrained solver instead");
    }

    if (problem.hasEqualityConstraints()) {
        if (!problem.jacobian_equality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of equality constraints must be provided");
        }
        equality_constraint_func_ = problem.equality_constraint_func.value();
        jacobian_equality_constraint_func_ = problem.jacobian_equality_constraint_func.value();
    }

    if (problem.hasInequalityConstraints()) {
        if (!problem.jacobian_inequality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of inequality constraints must be provided");
        }
        inequality_constraint_func_ = problem.inequality_constraint_func.value();
        jacobian_inequality_constraint_func_ = problem.jacobian_inequality_constraint_func.value();
    }

    cost_func_ = [&problem](const Eigen::VectorXd& x) { return problem.cost_func(x); };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) { return compute_gradient(problem, x); };

    // Initialize the approximate Hessian strategy based on the selected method
    auto strategy = BFGSHessianApproximation(problem);
    get_approximate_hessian_func_ = [strategy](const Eigen::VectorXd& grad_lagrangian, const Eigen::VectorXd& x) mutable {
        return strategy.getApproximateHessian(grad_lagrangian, x);
    };
};

ConstrainedSolver::ConstrainedSolver(const ConstrainedSolverOptions& options, const LSProblem& problem)
    : options_(std::cref(options)), x0_(problem.x0){

    if (!problem.hasEqualityConstraints() && !problem.hasInequalityConstraints()) {
        spdlog::warn("No constraints provided for the problem, call unconstrained solver instead");
    }

    if (problem.hasEqualityConstraints()) {
        if (!problem.jacobian_equality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of equality constraints must be provided");
        }
        equality_constraint_func_ = problem.equality_constraint_func.value();
        jacobian_equality_constraint_func_ = problem.jacobian_equality_constraint_func.value();
    }

    if (problem.hasInequalityConstraints()) {
        if (!problem.jacobian_inequality_constraint_func.has_value()) {
            throw std::invalid_argument("Jacobian of inequality constraints must be provided");
        }
        inequality_constraint_func_ = problem.inequality_constraint_func.value();
        jacobian_inequality_constraint_func_ = problem.jacobian_inequality_constraint_func.value();
    }

    cost_func_ = [&problem](const Eigen::VectorXd& x) {
        return problem.residual_func(x).transpose() * problem.residual_func(x);
    };
    gradient_func_ = [&problem](const Eigen::VectorXd& x) {
        if (problem.gradient_residual_func.has_value()) {
            return 2 * problem.gradient_residual_func.value()(x)* problem.residual_func(x);
        } else {
            // Numerical approximation of the Jacobian matrix of the residual function
            throw std::invalid_argument("Gradient of residual function must be provided");
        }
    };

    // Initialize the approximate Hessian strategy for least squares problems, which is always Gauss-Newton
    auto strategy = GaussNewtonHessianApproximation(problem);
    get_approximate_hessian_func_ = [strategy](const Eigen::VectorXd& grad_lagrangian, const Eigen::VectorXd& x) mutable {
        return strategy.getApproximateHessian(x);
    };
};

Result ConstrainedSolver::solve(){

    //The problem min_wrt_x  f(x)
    //                   s.t g(x)=0
    //                       h(x)>0

    //has resulting kkt conditions, satisfied at the optimimum(x', lambda', mhu'):
    //grad_L(x', lambda', mhu') = grad_f(x') - grad_g(x')*lambda' - grad_h(x')*mhu' = 0
    //g(x')=0
    //h(x')>0
    //mhu'>=0
    //h(x')^T * mhu' = 0

    //The newton step for those kkt conditions is the same as following QP:
    //min_wrt_(p_k) grad_f(x_k)^T*p_k + 0.5*p_k^T*Hessian_L(x_k, lambda_k, mhu_k)*p_K
    //  s.t         grad_g(x_k)^T*p_k + g(x_k)=0
    //              grad_h(x_k)^T*p_k + h(x_k)>=0

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = cost_func_(x0_);

    int iter = 0;
    double Dx_i = std::numeric_limits<double>::infinity();
    double Df_i = std::numeric_limits<double>::infinity();

    const int num_equality_constraints = jacobian_equality_constraint_func_ ? jacobian_equality_constraint_func_(x0_).rows() : 0;
    const int num_inequality_constraints = jacobian_inequality_constraint_func_ ? jacobian_inequality_constraint_func_(x0_).rows() : 0;
    Eigen::VectorXd lambda_i = Eigen::VectorXd::Zero(num_equality_constraints);
    Eigen::VectorXd mhu_i = Eigen::VectorXd::Zero(num_inequality_constraints);
    Eigen::VectorXd sigma_j = Eigen::VectorXd::Ones(num_equality_constraints);
    Eigen::VectorXd tau_j = Eigen::VectorXd::Ones(num_inequality_constraints);
    Eigen::VectorXd x_i = x0_;

    while (iter < options_.get().max_iter) {
        Eigen::VectorXd grad_f = gradient_func_(x_i);
        Eigen::MatrixXd grad_eq = jacobian_equality_constraint_func_(x_i);
        Eigen::MatrixXd grad_ineq = jacobian_inequality_constraint_func_(x_i);

        Eigen::VectorXd grad_lagrangian = grad_f - grad_eq*lambda_i - grad_ineq*mhu_i;

        if (Dx_i <= options_.get().step_tolerance) {
            result.summary.termination_reason = TerminationReason::StepTolerance;
            break;
        }
        if (Df_i <= options_.get().function_tolerance)
        {
            result.summary.termination_reason = TerminationReason::FunctionTolerance;
            break;
        }

        QPProblem Eq_qp_problem;
        // Eq_qp_problem.x0 = x_i;
        Eq_qp_problem.c = grad_f;
        Eq_qp_problem.H = get_approximate_hessian_func_(grad_lagrangian, x_i);
        Eq_qp_problem.A = grad_eq.transpose();
        Eq_qp_problem.b = equality_constraint_func_(x_i);
        Eq_qp_problem.C = grad_ineq.transpose();
        Eq_qp_problem.d = inequality_constraint_func_(x_i);

        QPSolver solver(options_.get().QP_subproblem_options, Eq_qp_problem);

        Result result = solver.solve();

        Eigen::VectorXd p_i = result.x;
        Eigen::VectorXd D_lambda_i = result.lambda - lambda_i;
        Eigen::VectorXd D_mhu_i = result.mhu - mhu_i;
        
        if ((grad_lagrangian*p_i).norm() < options_.get().gradient_tolerance) {
            result.summary.converged = true;
            result.summary.termination_reason = TerminationReason::GradientTolerance;
            break;
        }

        //Line search with armijo stopping condition on L1-Norm merit function
        double step_length = compute_step_length(options_.get().globalization_method, 
                                                 cost_func_,
                                                 equality_constraint_func_, 
                                                 inequality_constraint_func_, 
                                                 gradient_func_, 
                                                 jacobian_inequality_constraint_func_,
                                                 x_i,
                                                 p_i, 
                                                 sigma_j, 
                                                 tau_j);

        //Update sigma_j and tau_j
        for (int i = 0; i < sigma_j.size(); i++)
        {
            double abs_lambda_i = std::abs(lambda_i(i));
            sigma_j(i)=std::max(abs_lambda_i, (sigma_j(i)+abs_lambda_i)/2);
        }
        for (int i = 0; i < tau_j.size(); i++)
        {
            double abs_mhu_i = std::abs(mhu_i(i));
            tau_j(i)=std::max(abs_mhu_i, (tau_j(i)+abs_mhu_i)/2);
        }

        Eigen::VectorXd x_new = x_i + step_length * p_i;
        lambda_i = lambda_i + step_length * D_lambda_i;
        mhu_i = mhu_i + step_length * D_mhu_i;

        Dx_i = (x_new - x_i).norm()/std::max(x_i.norm(), 1e-16);
        double f_i = cost_func_(x_i);
        Df_i = std::abs(cost_func_(x_new) - f_i)/std::max(std::abs(f_i), 1e-16);

        x_i = x_new;
        iter++;
    }

    result.summary.iterations = iter;
    result.summary.final_cost = cost_func_(x_i);
    result.summary.final_gradient_norm = gradient_func_(x_i).norm();
    result.x = x_i;
    result.lambda = lambda_i;
    result.mhu = mhu_i;
    bool eq_constraint_satisfied = equality_constraint_func_(x_i).lpNorm<Eigen::Infinity>() <= options_.get().function_tolerance; //Here should go constraint tolerance
    bool inequality_constraint_satisfied = inequality_constraint_func_(x_i).lpNorm<Eigen::Infinity>() <= options_.get().function_tolerance; //Here should go constraint tolerance
    result.summary.converged = iter < options_.get().max_iter && eq_constraint_satisfied && inequality_constraint_satisfied;

    return result;
};

}