#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>

#include "solvers/unconstrained_solver.hpp"
#include "solvers/qp_solver.hpp"
#include "solvers/lp_solver.hpp"
#include "solvers/constrained_solver.hpp"
#include "direction_strategy.hpp"

#include <iostream>

using namespace furiaopt;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Catch::Approx;

// -----------------------------------------------------------------------------
//  helpers
// -----------------------------------------------------------------------------
static UnconstrainedSolverOptions nlp_opts(DirectionMethod m, int max_iter = 1000)
{
    UnconstrainedSolverOptions o;
    o.direction_method     = m;
    o.globalization_method = GlobalizationMethod::LineSearch;
    o.max_iter             = max_iter;
    o.gradient_tolerance   = 1e-8;
    o.step_tolerance       = 1e-14;
    o.function_tolerance   = 1e-16;
    //logger is initialized to null logger by default
    return o;
}

static IPMSolverOptions ipm_opts()
{
    return IPMSolverOptions{ /*tau_initial*/ 1.0, /*tau_factor*/ 0.2,
                             /*max_outer*/ 60, /*max_inner*/ 60, /*ipm_tol*/ 1e-9 };
}

// =============================================================================
//  Direction strategies
// =============================================================================
TEST_CASE("GradientDescentDirection returns -gradient", "[direction][gd]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);
    p.cost_func     = [](const VectorXd& x) -> double { return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x) -> VectorXd { return (2 * x).eval(); };

    GradientDescentDirection dir(p);
    VectorXd g(2); g << 2.0, 4.0;
    VectorXd d = dir.getDirection(g, VectorXd::Zero(2));
    REQUIRE(d.isApprox(-g, 1e-12));
}

TEST_CASE("ExactHessianDirection evaluates Hessian at current x_i", "[direction][newton]")
{
    // f(x) = 0.5 (x-3)^2 : H = 1, gradient at x_i = 5 is 2 -> Newton step = -2
    NLPProblem p;
    p.x0 = VectorXd::Constant(1, 0.0);
    const double target = 3.0;
    p.cost_func     = [target](const VectorXd& x){ return 0.5 * (x[0]-target)*(x[0]-target); };
    p.gradient_func = [target](const VectorXd& x){ VectorXd g(1); g[0]=x[0]-target; return g; };
    p.hessian_func  = [](const VectorXd&){ return MatrixXd::Identity(1,1); };

    ExactHessianDirection strat(p);
    VectorXd x_i(1); 
    x_i << 5.0;
    VectorXd g(1);   
    g << 2.0;
    VectorXd d = strat.getDirection(g, x_i);
    REQUIRE(d[0] == Approx(-2.0).epsilon(1e-10));
}

TEST_CASE("BFGS first call returns the steepest-descent direction", "[direction][bfgs]")
{
    NLPProblem p;
    p.x0 = VectorXd::Zero(2);
    p.cost_func     = [](const VectorXd& x){ return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x){ return (2*x).eval(); };
    BFGSDirection strat(p);
    VectorXd g(2); 
    g << 2.0, 4.0;
    VectorXd d = strat.getDirection(g, VectorXd::Zero(2));   // H0 = I -> d = -g
    REQUIRE(d.isApprox(-g, 1e-12));
}

// =============================================================================
//  Unconstrained QP (direct solve)
// =============================================================================
TEST_CASE("Unconstrained QP has the closed-form solution H x = -c", "[qp][unconstrained]")
{
    // min 0.5*(4*x0^2 + 2*x1^2) - 8*x0 - 6*x1
    // analytic solution: x = (2,3)
    QPProblem qp;
    qp.H = (MatrixXd(2,2) << 4,0, 0,2).finished(); //I use this advanced eigen initialization method to avoid constructing a temp matrix H just to assign qp.H
    qp.c = (VectorXd(2)   << -8,-6).finished();
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);          // no constraints -> no_constraints_QP_solver
    Result r = s.solve();

    VectorXd expected(2); expected << 2.0, 3.0;   // x = [2, 3]
    REQUIRE(r.x.isApprox(expected, 1e-9));
    REQUIRE(r.summary.converged);
}

TEST_CASE("Equality-constrained QP matches the KKT solution", "[qp][equality]")
{
    // min 0.5||x||^2  s.t.  x0 + x1 = 2  (written as A*x + b = 0 with b = -2)
    // analytic solution: x = (1,1), lambda from stationarity.
    QPProblem qp;
    qp.H = MatrixXd::Identity(2,2);
    qp.c = VectorXd::Zero(2);
    qp.A = (MatrixXd(1,2) << 1,1).finished();
    qp.b = (VectorXd(1)   << -2).finished();
    qp.x0 = VectorXd::Zero(2);

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r = s.solve();
    VectorXd expected(2); expected << 1.0, 1.0;
    REQUIRE(r.x.isApprox(expected, 1e-8));
    REQUIRE(r.summary.converged);
}

TEST_CASE("Inequality-constrained QP: inactive constraints", "[qp][inequality]")
{
    // min 0.5*xᵀHx + cᵀx
    //
    // H = [4 1]
    //     [1 2]
    //
    // c = [-8, -3]ᵀ
    //
    // subject to Cx + d >= 0, where
    //
    // C = [ 1  1]
    //     [-1  2]
    //     [ 1  0]
    //
    // d = [-1, 2, 0]ᵀ
    //
    // The unconstrained minimizer is x = [13/7, 4/7]ᵀ, which already satisfies
    // all inequalities strictly. Therefore the constrained optimum is also
    // x = [13/7, 4/7]ᵀ and none of the constraints is active.

    QPProblem qp;
    qp.H = (MatrixXd(2,2) << 4, 1, 1, 2).finished();
    qp.c = (VectorXd(2) << -8,-3).finished();
    qp.C = (MatrixXd(3,2) <<  1, 1, -1, 2, 1, 0).finished();
    qp.d = (VectorXd(3) << -1, 2, 0).finished();

    qp.x0 = (VectorXd(2) << 1.0, 1.0).finished(); // strictly feasible

    IPMSolverOptions ipm_options = ipm_opts();
    QPSolver s(ipm_options, qp);
    Result r = s.solve();

    VectorXd expected(2);
    expected << 13.0/7.0, 4.0/7.0;

    REQUIRE(r.x.isApprox(expected, 1e-4));

    // Check feasibility: C*x + d >= 0
    if (qp.C.has_value() && qp.d.has_value()) {
        REQUIRE(((*qp.C) * r.x + *qp.d).minCoeff() >= -1e-6);
    }
}

// =============================================================================
//  Unconstrained NLP solver
// =============================================================================
TEST_CASE("Exact Newton solves a quadratic in one step", "[solver][newton]")
{
    // QP problem with H = diag(2,3), c = [0,0]^T, optimum is obviously at x = [0,0]^T.
    NLPProblem p;
    p.x0 = (VectorXd(2) << 5.0, -4.0).finished();
    Eigen::Matrix2d H; 
    H << 2,0, 0,3;
    p.cost_func     = [H](const VectorXd& x){ return 0.5 * x.transpose() * H * x; };
    p.gradient_func = [H](const VectorXd& x){ return (H * x).eval(); };
    p.hessian_func  = [H](const VectorXd&){ return MatrixXd(H); };
    
    auto o = nlp_opts(DirectionMethod::ExactNewton);
    UnconstrainedSolver s(o, p);
    Result r = s.solve();

    REQUIRE(r.x.norm() < 1e-8);
    REQUIRE(r.summary.iterations <= 2);
    
}

TEST_CASE("BFGS converges on a strictly convex quadratic", "[solver][bfgs]")
{
    // f(x) = 1/2 xᵀHx + cᵀx
    //
    // H = [4 1]
    //     [1 3]
    //
    // c = [-1, -2]ᵀ
    //
    // H is symmetric positive definite, so the objective has a unique global
    // minimizer at x* = -H⁻¹c = [1/11, 7/11]ᵀ.

    NLPProblem p;
    p.x0 = (VectorXd(2) << 3.0, -2.0).finished();
    Eigen::Matrix2d H;
    H << 4, 1, 1, 3;
    Eigen::Vector2d c;
    c << -1, -2;
    p.cost_func = [H,c](const VectorXd& x) { return 0.5 * x.dot(H * x) + c.dot(x); };

    p.gradient_func = [H,c](const VectorXd& x) { return H * x + c; };

    auto options = nlp_opts(DirectionMethod::BFGS, 100);
    UnconstrainedSolver s(options, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << 1.0/11.0, 7.0/11.0;

    REQUIRE(r.x.isApprox(expected, 1e-6));
    REQUIRE(r.summary.final_cost < -0.68);
}

TEST_CASE("Gauss-Newton converges on an overdetermined nonlinear least-squares problem (m != n)",
          "[solver][gauss_newton][regression]")
{
    // Solve min 0.5 * ||F(x)||^2
    //
    // Residual:
    //
    //   F(x) =
    //   [ x0 - 1        ]
    //   [ x1 - 2        ]
    //   [ x0 + x1 - 3   ]
    //
    // There are m = 3 residuals and n = 2 variables.
    //
    // The minimum is exactly at:
    //   x* = [1, 2]
    //
    // because all residuals become zero:
    //
    //   F([1,2]) = [0,0,0].

    LSProblem p;
    p.x0 = (VectorXd(2) << -2.0, 5.0).finished();

    p.residual_func = [](const VectorXd& x) {
        VectorXd r(3);
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        r[2] = x[0] + x[1] - 3.0;
        return r;
    };

    p.gradient_residual_func = [](const VectorXd&) {
        // Transposed gradient Jᵀ (n x m) of the residual vector F(x):
        //
        // Jᵀ =
        // [ 1 0 1 ]
        // [ 0 1 1 ]

        MatrixXd JT(2, 3);
        JT << 1.0, 0.0, 1.0,
              0.0, 1.0, 1.0;
        return JT;
    };

    auto o = nlp_opts(DirectionMethod::GradientDescent, 100); //Gradient descent is correct as the problem will be seen as an NLP
    UnconstrainedSolver s(o, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << 1.0, 2.0;
    REQUIRE(r.x.isApprox(expected, 1e-6));
    REQUIRE(r.summary.final_cost < 1e-12);
    REQUIRE(r.summary.final_gradient_norm < 1e-6);
    REQUIRE(r.summary.converged);
}

// =============================================================================
//  Convergence-flag / termination-reason semantics 
// =============================================================================
TEST_CASE("Gradient-norm termination reports GradientTolerance", "[solver][termination][regression]")
{
    // Regression for the convergence test scaling: a gradient-tolerance stop
    // must leave final ||g|| <= gradient_tolerance (not ||g||^2).
    NLPProblem p;
    p.x0 = (VectorXd(2) << 3.0, -2.0).finished();
    p.cost_func     = [](const VectorXd& x){ return x.squaredNorm(); };
    p.gradient_func = [](const VectorXd& x){ return (2*x).eval(); };

    auto o = nlp_opts(DirectionMethod::GradientDescent);
    o.gradient_tolerance = 1e-6;
    UnconstrainedSolver s(o, p);
    Result r = s.solve();
    REQUIRE(r.summary.converged);
    REQUIRE(r.summary.termination_reason == TerminationReason::GradientTolerance);
    REQUIRE(r.summary.final_gradient_norm <= 1e-6);
}

TEST_CASE("Function-tolerance check measures relative COST CHANGE", "[solver][termination][regression]")
{
    // Regression for the `- 000` typo: with a nonzero-optimum-cost problem,
    // a function-tolerance stop must be triggered by |f_new - f_old|, not |f_new|.
    // f(x) = (x-2)^2 + 10  (optimum cost = 10, far from 0).
    NLPProblem p;
    p.x0 = VectorXd::Constant(1, 5.0);
    p.cost_func     = [](const VectorXd& x){ return (x[0]-2)*(x[0]-2) + 10.0; };
    p.gradient_func = [](const VectorXd& x){ VectorXd g(1); g[0]=2*(x[0]-2); return g; };

    auto o = nlp_opts(DirectionMethod::GradientDescent, 100000);
    UnconstrainedSolver s(o, p);
    Result r = s.solve();
    REQUIRE(r.x[0] == Approx(2.0).margin(1e-3));   // must reach the true minimiser
}

// =============================================================================
//  SQP / ConstrainedSolver 
// =============================================================================
static ConstrainedSolverOptions con_opts()
{
    ConstrainedSolverOptions o;
    o.direction_method     = DirectionMethod::BFGS;
    o.globalization_method = GlobalizationMethod::LineSearch;
    o.max_iter             = 100;
    o.gradient_tolerance   = 1e-7;
    o.step_tolerance       = 1e-12;
    o.function_tolerance   = 1e-14;
    // o.logger is already initialized with null logger by default
    o.QP_subproblem_options = ipm_opts();
    return o;
}

TEST_CASE("SQP: quadratic objective with linear equality constraint", "[sqp][equality]")
{
    // min 0.5*(x0^2 + x1^2)
    //
    // s.t.
    //      x0 + x1 - 2 = 0
    //
    // The solution is the projection of the origin onto the line x0+x1=2:
    //
    //      x* = [1,1]
    //
    // Objective value:
    //      f(x*) = 1

    NLPProblem p;

    p.x0 = (VectorXd(2) << 0.0, 0.0).finished();

    p.cost_func = [](const VectorXd& x) {
        return 0.5 * x.squaredNorm();
    };

    p.gradient_func = [](const VectorXd& x) {
        return x;
    };

    p.equality_constraint_func = [](const VectorXd& x) {
        VectorXd c(1);
        c[0] = x[0] + x[1] - 2.0;
        return c;
    };

    p.gradient_equality_constraint_func = [](const VectorXd&) {
        MatrixXd JT(2,1);
        JT << 1.0,
             1.0;
        return JT;
    };

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << 1.0, 1.0;
    
    CAPTURE(r.summary.iterations, r.summary.final_cost, r.x.transpose());
    REQUIRE(r.x.isApprox(expected, 1e-6));
    
}

TEST_CASE("SQP: quadratic objective with active inequality constraint", "[sqp][inequality]")
{
    // min 0.5*((x0-2)^2 + (x1-2)^2)
    //
    // s.t.
    //      x0 + x1 <= 2
    //
    // written as:
    //      2 - x0 - x1 >= 0
    //
    // The unconstrained solution is (2,2), which violates the constraint.
    //
    // The constrained optimum is the projection onto x0+x1=2:
    //
    //      x* = [1,1]

    NLPProblem p;

    p.x0 = (VectorXd(2) << 0.0, 0.0).finished();

    p.cost_func = [](const VectorXd& x) {
        return 0.5 * ((x[0]-2)*(x[0]-2) +
                      (x[1]-2)*(x[1]-2));
    };

    p.gradient_func = [](const VectorXd& x) {
        VectorXd g(2);
        g << x[0]-2, x[1]-2;
        return g;
    };

    p.inequality_constraint_func = [](const VectorXd& x) {
        VectorXd c(1);
        c[0] = 2.0 - x[0] - x[1];
        return c;
    };

    p.gradient_inequality_constraint_func = [](const VectorXd&) {
        MatrixXd JT(2,1);
        JT << -1.0,
             -1.0;
        return JT;
    };

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << 1.0, 1.0;

    REQUIRE(r.x.isApprox(expected, 1e-5));
}

TEST_CASE("SQP: nonlinear equality constraint circle", "[sqp][nonlinear]")
{
    // min x0 + x1
    //
    // s.t.
    //      x0^2 + x1^2 - 1 = 0
    //
    // The feasible set is the unit circle.
    //
    // The minimum of x0+x1 occurs at:
    //
    //      x* = [-sqrt(2)/2, -sqrt(2)/2]

    NLPProblem p;

    p.x0 = (VectorXd(2) << 0.8, 0.2).finished();

    p.cost_func = [](const VectorXd& x) {
        return x[0] + x[1];
    };

    p.gradient_func = [](const VectorXd&) {
        VectorXd g(2);
        g << 1.0, 1.0;
        return g;
    };

    p.equality_constraint_func = [](const VectorXd& x) {
        VectorXd c(1);
        c[0] = x[0]*x[0] + x[1]*x[1] - 1.0;
        return c;
    };

    p.gradient_equality_constraint_func = [](const VectorXd& x) {
        MatrixXd JT(2,1);
        JT << 2*x[0], 2*x[1];
        return JT;
    };

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << -std::sqrt(2)/2,
                -std::sqrt(2)/2;

    CAPTURE(r.x);
    CAPTURE(r.summary.converged);
    CAPTURE(r.summary.termination_reason);
    CAPTURE(r.summary.iterations);

    REQUIRE(r.x.isApprox(expected, 1e-4));
}

TEST_CASE("SQP: nonlinear equality with active inequality", "[sqp][mixed]")
{
    // min (x0-1)^2 + (x1-1)^2
    //
    // s.t.
    //      x0^2 + x1^2 = 1
    //      x0 >= 0
    //
    // The closest point on the first quadrant of the unit circle
    // to (1,1) is:
    //
    //      x* = [sqrt(2)/2, sqrt(2)/2]

    NLPProblem p;

    p.x0 = (VectorXd(2) << 0.2, 0.9).finished();

    p.cost_func = [](const VectorXd& x) {
        return (x[0]-1)*(x[0]-1) +
               (x[1]-1)*(x[1]-1);
    };

    p.gradient_func = [](const VectorXd& x) {
        VectorXd g(2);
        g << 2*(x[0]-1),
             2*(x[1]-1);
        return g;
    };

    p.equality_constraint_func = [](const VectorXd& x) {
        VectorXd c(1);
        c[0] = x.squaredNorm() - 1.0;
        return c;
    };

    p.gradient_equality_constraint_func = [](const VectorXd& x) {
        MatrixXd JT(2,1);
        JT << 2*x[0],
             2*x[1];
        return JT;
    };

    p.inequality_constraint_func = [](const VectorXd& x) {
        VectorXd c(1);
        c[0] = x[0];
        return c;
    };

    p.gradient_inequality_constraint_func = [](const VectorXd&) {
        MatrixXd JT(2,1);
        JT << 1.0,
             0.0;
        return JT;
    };

    auto o = con_opts();
    ConstrainedSolver s(o, p);
    Result r = s.solve();

    VectorXd expected(2);
    expected << std::sqrt(2)/2,
                std::sqrt(2)/2;

    REQUIRE(r.x.isApprox(expected, 1e-4));
}

