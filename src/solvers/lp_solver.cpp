#include "solvers/lp_solver.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include "generalization_method.hpp"

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

    if (problem_.get().hasInequalityConstraints() || problem_.get().hasEqualityConstraints())
    {
        general_LP_solver(result);
    }
    else
    {
        spdlog::error("No constraints LP solver does not make sense");
        throw std::runtime_error("No constraints LP solver does not make sense");
    }

    return result;
};

void LPSolver::general_LP_solver(Result& result)
{   
    //The LP problem is solved using the interior point method.
    //The problem is reformulated using barrier functions:
    //min c^T*x - tau*sum_i(log(C*x_i + d_i))
    //s.t Ax + b = 0

    //KKT conditions for this problem reads as follows:
    //c - A^T*lambda - tau*sum_i(C_i/(C*x_i + d_i)) = 0
    //Ax + b = 0

    //Usign Newton's method to solve the KKT conditions, we get the following update equations:
    // [ tau*C^T*(diag(C*x + d)^-2)*C | -A^T ; * [ dx     ;   = [ -c + A^T*lamnda + tau*C^T*(diag(C*x + d)^-1)*Eigen::VectorXd::Ones(c.rows());
    //   A                            |  0   ]    dlambda ]       -Ax - b                                                                     ]

    // Extract problem data safely
    const Eigen::VectorXd& c = problem_.get().c;
    const Eigen::MatrixXd& A = problem_.get().A.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& b = problem_.get().b.value_or(Eigen::VectorXd::Zero(0));
    const Eigen::MatrixXd& C = problem_.get().C.value_or(Eigen::MatrixXd::Zero(0, c.rows()));
    const Eigen::VectorXd& d = problem_.get().d.value_or(Eigen::VectorXd::Zero(0));

    // Dimensions
    const size_t n = c.rows();
    const size_t m_eq = A.rows();

    // Initialize variables
    Eigen::VectorXd x = problem_.get().x0;
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_eq);

    // IPM Hyperparameters
    double tau = 10.0;          // Initial barrier parameter
    const double mu = 0.1;      // Tau reduction factor (tau = tau * mu)
    const int max_outer = 20;   // Outer iterations (centering steps)
    const int max_inner = 30;   // Inner Newton iterations
    const double tol = 1e-6;    // KKT tolerance

    // Pre-allocate KKT System: Left-Hand Side (LHS) and Right-Hand Side (RHS)
    Eigen::MatrixXd KKT = Eigen::MatrixXd::Zero(n + m_eq, n + m_eq);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + m_eq);

    // Top-right and Bottom-left blocks of KKT are constant throughout the loop
    KKT.block(0, n, n, m_eq) = -A.transpose();
    KKT.block(n, 0, m_eq, n) = A;

    for (int outer = 0; outer < max_outer; ++outer) {
        for (int inner = 0; inner < max_inner; ++inner) {
            
            // Vector elements: s_i = C_i * x + d_i
            Eigen::VectorXd s = C * x + d;
            
            // Check strictly feasible condition safety boundary
            if ((s.array() <= 0).any()) {
                throw std::runtime_error("IPM drifted out of the feasible region (s <= 0).");
            }

            // Compute inverse scaling matrices entries
            Eigen::VectorXd s_inv = s.cwiseInverse();
            Eigen::VectorXd s_inv2 = s_inv.cwiseAbs2();

            // 1. Compute LHS Hessian component: tau * C^T * diag(s)^-2 * C
            // Using .asDiagonal() avoids full matrix multiplication overhead
            KKT.block(0, 0, n, n) = tau * C.transpose() * s_inv2.asDiagonal() * C;

            // 2. Compute RHS Residuals
            Eigen::VectorXd dual_res = -c + A.transpose() * lambda + tau * C.transpose() * s_inv;
            Eigen::VectorXd primal_res = -A * x - b;

            rhs.head(n) = dual_res;
            rhs.tail(m_eq) = primal_res;

            // Check Convergence
            if (dual_res.norm() < tol && primal_res.norm() < tol) {
                break;
            }

            // 3. Solve the full KKT system using ColPivHouseholderQR (robust for dense KKTs)
            auto qr = KKT.colPivHouseholderQr();

            if (qr.rank() < KKT.rows())
            {
                spdlog::error("KKT matrix is rank deficient. IPM cannot proceed.");
                throw std::runtime_error("KKT matrix is rank deficient. IPM cannot proceed.");
            }

            Eigen::VectorXd delta = qr.solve(rhs);
            Eigen::VectorXd dx = delta.head(n);
            Eigen::VectorXd dlambda = delta.tail(m_eq);

            // 4. Backtracking Line Search (Ensure inequality constraints are never violated)
            double alpha = 1.0;
            const double beta = 0.5; // Step reduction factor
            
            // Fraction-to-the-boundary rule: ensure C*(x + alpha*dx) + d > 0
            while (alpha > 1e-8) {
                Eigen::VectorXd s_next = C * (x + alpha * dx) + d;
                if ((s_next.array() > 0).all()) {
                    break;
                }
                alpha *= beta;
            }

            // Update positions
            x += alpha * dx;
            lambda += alpha * dlambda;

            if (dx.norm() < options_.get().step_tolerance) break;
        }

        // Reduce barrier parameter to tighten the approximation to the true LP
        tau *= mu;
        if (tau < 1e-8) break;
    }

    // Pack results
    result.x = x;
    result.summary.final_cost = c.transpose() * x;
    result.summary.converged = true;

};

}