#include "solvers/qp_solver.hpp"
#include <iostream>
#include <Eigen/Dense>

// Config loader
#include "config_loader.hpp"

// Logger
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>

void init_logger(const std::string& log_file_folder)
{
    auto logger = spdlog::basic_logger_mt(
        "furia_optimizer_logger",
        log_file_folder + "qp_balance.log",
        true
    );
    logger->set_level(spdlog::level::info);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    spdlog::set_default_logger(logger);
}

int main()
{
    furiaoptimizer::SolverOptions options = furiaoptimizer::load_solver_options("config/config.json");
    init_logger(options.log_file_folder_path);

    // Problem Constants
    double mass = 12.0;       // 12 kg robot dog
    double g = 9.81;          // Gravity
    double F_g = mass * g;    // Total downward force (~117.7 N)
    double mu = 0.6;          // Friction coefficient of the ground

    furiaoptimizer::QPProblem problem;

    // --- 1. Objective Function ---
    // Minimize 1/2 * u^T * H * u + c^T * u
    // u = [f_x1, f_z1, f_x2, f_z2]^T
    // Identity matrix minimizes the raw force magnitude across all legs evenly.
    problem.H = Eigen::MatrixXd::Identity(4, 4);
    // Simulate an external 20N force pushing the robot along the X-axis.
    // To counteract it, the sum of foot forces must handle this shift.
    // We apply a linear cost bias to simulate the directional disturbance:
    problem.c = Eigen::VectorXd::Zero(4);
    problem.c << -10.0, 0.0, -10.0, 0.0; // Biases the solver to expect horizontal strain

    // --- 2. Equality Constraints (Force Balance) ---
    // Row 0 (Horizontal balance): f_x1 + f_x2 = 0
    // Row 1 (Vertical balance):   f_z1 + f_z2 = F_g
    // Standard Form: A*u + b = 0  => A*u - [0, F_g]^T = 0
    Eigen::MatrixXd A_eq(2, 4);
    A_eq << 1.0, 0.0, 1.0, 0.0,   // f_x1 + f_x2
            0.0, 1.0, 0.0, 1.0;   // f_z1 + f_z2
    problem.A = A_eq;

    Eigen::VectorXd b_eq(2);
    b_eq <<-20.0, -F_g;
    problem.b = b_eq;

    // --- 3. Inequality Constraints (Friction Cones) ---
    // For each foot, the horizontal force must be bounded by friction:
    // -mu * f_z <= f_x <= mu * f_z
    // This gives two inequalities per foot:
    // 1)  mu * f_z - f_x >= 0
    // 2)  mu * f_z + f_x >= 0
    // Standard Form: C*u + d >= 0
    Eigen::MatrixXd C_ineq(4, 4);
    C_ineq.setZero();
    // Foot 1
    C_ineq(0, 0) = -1.0;  C_ineq(0, 1) = mu;   // -f_x1 + mu*f_z1 >= 0
    C_ineq(1, 0) =  1.0;  C_ineq(1, 1) = mu;   //  f_x1 + mu*f_z1 >= 0
    // Foot 2
    C_ineq(2, 2) = -1.0;  C_ineq(2, 3) = mu;   // -f_x2 + mu*f_z2 >= 0
    C_ineq(3, 2) =  1.0;  C_ineq(3, 3) = mu;   //  f_x2 + mu*f_z2 >= 0
    problem.C = C_ineq;

    problem.d = Eigen::VectorXd::Zero(4); // No static offsets needed

    // --- 4. Initial Feasible Guess ---
    // Start with legs pushing straight down sharing the weight equally, 0 horizontal force.
    Eigen::VectorXd u_init(4);
    u_init << 0.0, F_g / 2.0, 0.0, F_g / 2.0;
    problem.x0 = u_init;

    // 5. Solve
    furiaoptimizer::QPSolver solver(options, problem);
    furiaoptimizer::Result result = solver.solve();

    // 6. Visual Output Verification
    std::cout << "==================================================\n";
    std::cout << "        ROBOTIC STATIC FORCE ALLOCATION           \n";
    std::cout << "==================================================\n";
    std::cout << "Optimal Foot Forces Vector:\n" << result.x << "\n\n";
    std::cout << "Foot 1 -> Horizontal: " << result.x[0] << " N, Vertical: " << result.x[1] << " N\n";
    std::cout << "Foot 2 -> Horizontal: " << result.x[2] << " N, Vertical: " << result.x[3] << " N\n";
    std::cout << "--------------------------------------------------\n";
    
    // Check friction boundary ratio (|f_x| / f_z) -> must be <= mu (0.6)
    std::cout << "Foot 1 Friction Ratio (|f_x|/f_z): " << std::abs(result.x[0]) / result.x[1] << " (Limit: " << mu << ")\n";
    std::cout << "Foot 2 Friction Ratio (|f_x|/f_z): " << std::abs(result.x[2]) / result.x[3] << " (Limit: " << mu << ")\n";
    std::cout << "Converged: " << (result.summary.converged ? "Yes" : "No") << "\n";
    std::cout << "==================================================\n";

    return 0;
}