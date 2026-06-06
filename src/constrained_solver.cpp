#include "constrained_solver.hpp"
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

ConstrainedSolver::ConstrainedSolver(const SolverOptions& options, const Problem& problem){
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

Result ConstrainedSolver::solve(){

    spdlog::info("Starting solve");
    Result result;
    result.summary.initial_cost = problem_.cost_func(problem_.x0);

    if (problem_.isQuadratic() && problem_.hasOnlyLinearConstraints())
    {
        spdlog::info("Quadratic problem with only linear constraints detected, using direct solver");
        // For quadratic problems with only linear constraints, we can directly compute the optimal solution by solving the KKT conditions
        QP_solver(result, problem_);
    }
    else
    {
        spdlog::info("Quadratic problem detected");
    }
    else if (problem_.isLinear())
    {
        spdlog::info("Linear problem detected");
    }
    else
    {
        spdlog::info("Non-linear problem detected");
    }

    return result;
};

void ConstrainedSolver::SQP_solver(Result& result, const Problem& problem_){

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
}

void ConstrainedSolver::QP_solver(Result& result, const Problem& problem_){
    Eigen::VectorXd x_i = problem_.x0;
    int n = x_i.size();
    Eigen::MatrixXd H = problem_.H_quadratic.value();
    Eigen::VectorXd c = problem_.c_quadratic.value();
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    //Sizes are automatically handled by Eigen, we just need to make sure that the user provides consistent dimensions in the problem definition
    if (problem_.equality_constraint_func.has_value()) {
        A = problem_.jacobian_equality_constraint_func.value()(x_i);
        b = problem_.equality_constraint_func.value()(Eigen::VectorXd::Zero(x_i.size()));
    }
    if (!problem_.inequality_constraint_func.has_value()) {
        // KKT conditions for equality constrained QP can be solved by solving the linear system:
        // [H, -A'; A, 0] [x; lambda] = [-c; -b]
        int m_eq = A.rows();
        Eigen::MatrixXd KKT_matrix ( n + m_eq, n + m_eq);
        KKT_matrix << H, - A.transpose(),
                      A, Eigen::MatrixXd::Zero(m_eq, m_eq);
        Eigen::VectorXd rhs (n + m_eq);
        rhs << -c, -b;
        // Solve the KKT system using a direct solver, 
        // Following Eigen guidance we can use Eigen's FullPivLU decomposition
        Eigen::FullPivLU<Eigen::MatrixXd> lu(KKT_matrix);

        Eigen::VectorXd sol = lu.solve(rhs);
        Eigen::VectorXd x = sol.head(n);
        Eigen::VectorXd lambda = sol.tail(m_eq);
        result.x = x;
        result.summary.final_cost = problem_.cost_func(result.x);
        result.summary.iterations = 0;
        result.summary.converged = lu.isInvertible();
        return;
    }

    Eigen::MatrixXd C;
    Eigen::VectorXd d;
    if (problem_.inequality_constraint_func.has_value()) {
        C = problem_.jacobian_inequality_constraint_func.value()(x_i);
        d = problem_.inequality_constraint_func.value()(Eigen::VectorXd::Zero(x_i.size()));

        // Define Auxiliary LP problem to find a feasible starting point for the interior point method:
        // min sum(s)
        // s.t Ax_0 + b = 0
        //     Cx_0 + d - s = 0
        // s >= 0
        int problem_size = x_i.size() + d.size();
        Problem aux_problem;
        aux_problem.x0 = Eigen::VectorXd::Ones(problem_size); // Start with a strictly positive slack variable
        aux_problem.c_linear = Eigen::VectorXd::Ones(d.size()); // Objective is to minimize the sum of slacks
        aux_problem.cost_func = [d](const Eigen::VectorXd& z) {
            int n = z.size() - d.size(); // Original variable size
            Eigen::VectorXd s = z.tail(d.size());
            return s.sum();
        };

        Eigen::MatrixXd A_bar;
        A_bar << A, Eigen::MatrixXd::Zero(A.rows(), b.size()), 
                 C, - Eigen::MatrixXd::Ones(C.rows(), d.size());
        Eigen::VectorXd b_bar (A.rows() + C.rows());
        b_bar << +b, +d;
        Eigen::MatrixXd C_bar;
        C_bar << Eigen::MatrixXd::Zero(C.rows(), problem_size), 
                 Eigen::MatrixXd::Zero(C.rows(), C.cols()), Eigen::MatrixXd::Ones(d.size(), d.size());
        aux_problem.equality_constraint_func = [A_bar, b_bar](const Eigen::VectorXd& z) {
            return A_bar * z + b_bar;
        };
        aux_problem.jacobian_equality_constraint_func = [A_bar](const Eigen::VectorXd& z) {
            return A_bar;
        };
        aux_problem.inequality_constraint_func = [C_bar](const Eigen::VectorXd& z) {
            return C_bar * z;
        };
        aux_problem.jacobian_inequality_constraint_func = [C_bar](const Eigen::VectorXd& z) {
            return C_bar;
        };
        Result aux_result;
        LP_solver(aux_result, aux_problem);

        Eigen::VectorXd x0 = aux_result.x.head(x_i.size());

        // Now that we have a feasible starting point, we can solve the original problem using an interior point method.
        
    }

};

void ConstrainedSolver::LP_solver(Result& result, const Problem& problem_) {
    // Given a LP problem of the form:
    // min c^T x
    // s.t Ax + b = 0
    //     Cx + d >= 0 

    // 1. Setup Dimensions
    const int n = problem_.x0.size();
    const Eigen::MatrixXd A = problem_.jacobian_equality_constraint_func.value()(problem_.x0);
    const Eigen::VectorXd b = problem_.equality_constraint_func.value()(Eigen::VectorXd::Zero(n));
    const Eigen::MatrixXd C = problem_.jacobian_inequality_constraint_func.value()(problem_.x0);
    const Eigen::VectorXd d = problem_.inequality_constraint_func.value()(Eigen::VectorXd::Zero(n));
    const Eigen::VectorXd c = problem_.c_linear.value();

    const int m_eq = A.rows();
    const int m_ineq = C.rows();

    //Convert the linear inqueality constraints to barrier functions.
    //Define the slack variables s = C*x + d, the problem becomes:
    // min c^T x - mu * sum(log(s))
    // s.t Ax + b = 0
    //     s = Cx + d
    // Notice the constraint: s>0, is automatically enforced since as the s -> 0 , the - sum(log(s)) -> +inf

    // 2. Initialize Variables (Cold Start)
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s = Eigen::VectorXd::Ones(m_ineq); // Slacks must be > 0
    Eigen::VectorXd y = Eigen::VectorXd::Zero(m_eq); // Lagrange multipliers for equality constraints (can be any real number)
    Eigen::VectorXd z = Eigen::VectorXd::Ones(m_ineq); // Lagrange multipliers for inequality constraints (must be > 0)

    const int max_iter = 50;
    const double tol = 1e-8;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute Residuals
        Eigen::VectorXd rp = A * x + b;
        Eigen::VectorXd rd = C * x - s + d;
        Eigen::VectorXd r_dual = A.transpose() * y + C.transpose() * z + c;
        
        double duality_gap = s.dot(z);
        if (rp.norm() < tol && rd.norm() < tol && r_dual.norm() < tol && duality_gap < tol) {
            break;
        }

        // 3. Centrality parameter (mu)
        double mu = 0.1 * duality_gap / m_ineq;

        // 4. Solve Newton System (KKT System)
        // We use a condensed system to solve for (dx, dy)
        // Diag matrices
        Eigen::VectorXd invS = s.array().inverse();
        Eigen::MatrixXd Z_invS = (z.array() * invS.array()).matrix().asDiagonal();
        
        // Form the Augmented System Matrix:
        // [ -(C^T * Z/S * C)   A^T ] [ dx ] = [ rhs_x ]
        // [        A            0  ] [ dy ] = [ rhs_y ]
        
        Eigen::MatrixXd K_top_left = -C.transpose() * Z_invS * C;
        Eigen::MatrixXd K(n + m_eq, n + m_eq);
        K.setZero();
        K.block(0, 0, n, n) = K_top_left;
        K.block(0, n, n, m_eq) = A.transpose();
        K.block(n, 0, m_eq, n) = A;

        Eigen::VectorXd r_comp = (s.array() * z.array() - mu).matrix();
        Eigen::VectorXd rhs_x = -r_dual + C.transpose() * invS.asDiagonal() * (r_comp - z.asDiagonal() * rd);
        Eigen::VectorXd rhs_y = -rp;

        Eigen::VectorXd rhs(n + m_eq);
        rhs << rhs_x, rhs_y;

        // Solve via LDLT (K is symmetric indefinite)
        Eigen::VectorXd delta = K.ldlt().solve(rhs);
        Eigen::VectorXd dx = delta.head(n);
        Eigen::VectorXd dy = delta.tail(m_eq);

        // Recover ds and dz
        Eigen::VectorXd ds = C * dx + rd;
        Eigen::VectorXd dz = invS.asDiagonal() * (-r_comp - z.asDiagonal() * ds);

        // 5. Line Search (Fraction-to-the-boundary rule)
        auto get_step = [](const Eigen::VectorXd& v, const Eigen::VectorXd& dv) {
            double alpha = 1.0;
            for (int i = 0; i < v.size(); ++i) {
                if (dv(i) < 0) alpha = std::min(alpha, -0.99 * v(i) / dv(i));
            }
            return alpha;
        };

        double alpha_p = get_step(s, ds);
        double alpha_d = get_step(z, dz);

        // 6. Update
        x += alpha_p * dx;
        s += alpha_p * ds;
        y += alpha_d * dy;
        z += alpha_d * dz;
    }
    
    result.x = x;
    result.summary.final_cost = problem_.cost_func(result.x);
    result.summary.iterations = iter;
    result.summary.converged = iter < max_iter;
}

}