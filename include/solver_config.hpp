#pragma once
#include <Eigen/Dense>
#include <optional>
#include <memory>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>
#include "types.hpp"

namespace furiaopt
{

enum class DirectionMethod {
    GradientDescent,
    BFGS,
    ExactNewton
};

enum class GlobalizationMethod {
    LineSearch,
    TrustRegion
};
struct UnconstrainedSolverOptions {
    DirectionMethod direction_method;
    GlobalizationMethod globalization_method;
    int max_iter;
    double gradient_tolerance;
    double step_tolerance;
    double function_tolerance;
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>());
};

struct IPMSolverOptions {
    double tau_initial;
    double tau_factor;
    int max_outer;
    int max_inner;
    double ipm_tol;
    std::shared_ptr<spdlog::logger> logger = std::make_shared<spdlog::logger>("null", std::make_shared<spdlog::sinks::null_sink_mt>());
};

struct ConstrainedSolverOptions: UnconstrainedSolverOptions {
    IPMSolverOptions QP_subproblem_options; 
};

struct LinearConstraints {

    std::optional<Eigen::MatrixXd> A; // Coefficients for linear equality constraints
    std::optional<Eigen::VectorXd> b; // Constants for linear equality constraints
    std::optional<Eigen::MatrixXd> C; // Coefficients for linear inequality constraints
    std::optional<Eigen::VectorXd> d; // Constants for linear inequality constraints

    bool hasEqualityConstraints() const { return A.has_value() && b.has_value(); }
    bool hasInequalityConstraints() const { return C.has_value() && d.has_value(); }
};

// For Linear Programming: min f(x) = c^T*x
//                         s.t Ax + b = 0   (equality constraints)
//                         Cx + d >= 0      (inequality constraints)
struct LPProblem: public LinearConstraints{
    std::optional<Eigen::VectorXd> x0;
    Eigen::VectorXd c; // Linear term

};

// For Quadratic Programming: min f(x) = 0.5 * x'Hx + c'x
//                         s.t Ax + b = 0   (equality constraints)
//                         Cx + d >= 0      (inequality constraints)
struct QPProblem: public LinearConstraints {
    std::optional<Eigen::VectorXd> x0;
    Eigen::MatrixXd H; // Quadratic term
    Eigen::VectorXd c; // Linear term
};

struct NonLinearConstraints {
    // equality constraints: g(x) = 0
    // Notice linear equality constrainst are rapresented as g(x) = Ax + b = 0
    std::optional<EqualityConstraintFunc> equality_constraint_func;
    std::optional<JacobianEqualityConstraintFunc> jacobian_equality_constraint_func;

    // Inequality constraints: h(x) >= 0
    // Notice linear Inequality constrainst are rapresented as h(x) = Cx + d >= 0
    std::optional<InequalityConstraintFunc> inequality_constraint_func;
    std::optional<JacobianInequalityConstraintFunc> jacobian_inequality_constraint_func;

    bool hasEqualityConstraints() const { return equality_constraint_func.has_value(); }
    bool hasInequalityConstraints() const { return inequality_constraint_func.has_value(); }
};

//Use it for Gauss-Newton methods. The cost function is defined as f(x) = 0.5 * F(x)^T*F(x), where F(x) is a column vector of functions.
struct LSProblem : public NonLinearConstraints {
    Eigen::VectorXd x0;
    ResidualFunc residual_func;
    std::optional<GradientResidualFunc> gradient_residual_func;

    bool hasGradientResidualFunc() const { return gradient_residual_func.has_value(); }
};

//Use it for Newton/Quasi-Newton methods. where the cost function is a generic non linear function f(x).
struct NLPProblem : public NonLinearConstraints {
    
    Eigen::VectorXd x0;
    CostFunc cost_func;

    std::optional<GradientFunc> gradient_func;
    std::optional<HessianFunc> hessian_func;

    bool hasGradient() const { return gradient_func.has_value(); }
    bool hasHessian() const { return hessian_func.has_value(); }

};

enum class TerminationReason {
    MaxIterations,
    GradientTolerance,
    StepTolerance,
    FunctionTolerance
};

struct SolverSummary {
    int iterations = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double final_gradient_norm = 0.0;
    bool converged = false;
    TerminationReason termination_reason;
};

struct Result {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(0);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(0);
    Eigen::VectorXd mhu = Eigen::VectorXd::Zero(0);
    SolverSummary summary;
};

}