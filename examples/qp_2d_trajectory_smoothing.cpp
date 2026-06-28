#include "solvers/qp_solver.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>

// Config loader
#include "config_loader.hpp"

int main()
{
    // 1. Initialize Configuration and Logger
    furiaopt::IPMSolverOptions options = furiaopt::load_ipm_solver_options("config/config.json");

    options.logger->info("=================================================");
    options.logger->info("2D Trajectory Smoothing Optimization Pipeline Init");
    options.logger->info("=================================================");

    // 2. Structural Dimensions Setup
    // Total Optimization Variables = 100 -> implies 50 points for X and 50 points for Y
    const int points_per_dim = 50;
    const int N = points_per_dim - 1; // 49 intervals
    const int num_vars = 2 * points_per_dim; // 100 total decision variables

    options.logger->info("Total optimization variables allocated: {} ({} points for X, {} points for Y)",
                 num_vars, points_per_dim, points_per_dim);

    furiaopt::QPProblem problem;

    // --- 3. Objective Function: Minimize Acceleration in 2D ---
    // Acceleration Matrix D for 1D dimension tracking: size (N-1) x (N+1) -> 48 x 50
    Eigen::MatrixXd D_1d = Eigen::MatrixXd::Zero(N - 1, points_per_dim);
    for (int i = 0; i < N - 1; ++i) {
        D_1d(i, i)     = 1.0;   // coordinate_{k-1}
        D_1d(i, i + 1) = -2.0;  // -2 * coordinate_k
        D_1d(i, i + 2) = 1.0;   // coordinate_{k+1}
    }

    // Standard formulation: 1/2 * u^T * H * u where H is block diagonal
    Eigen::MatrixXd H_1d = 2.0 * D_1d.transpose() * D_1d;
    
    problem.H = Eigen::MatrixXd::Zero(num_vars, num_vars);
    problem.H.block(0, 0, points_per_dim, points_per_dim) = H_1d;                     // X block
    problem.H.block(points_per_dim, points_per_dim, points_per_dim, points_per_dim) = H_1d; // Y block
    problem.c = Eigen::VectorXd::Zero(num_vars);

    // --- 4. Equality Constraints Setup (4 Coordinates Fixed) ---
    // Anchors: Start (k=0), Midpoint A (k=15), Midpoint B (k=35), End (k=49)
    // Each 2D waypoint anchor requires 2 equations (1 for X, 1 for Y) -> 8 rows total
    const int num_eq_rows = 8;
    Eigen::MatrixXd A_eq = Eigen::MatrixXd::Zero(num_eq_rows, num_vars);
    Eigen::VectorXd b_eq = Eigen::VectorXd::Zero(num_eq_rows);

    // Define anchor values (Time-Space profiles)
    struct EqualityAnchor { int index; double x_val; double y_val; };
    std::vector<EqualityAnchor> eq_anchors = {
        {0,   0.0,  0.0},   // Start Position
        {15,  4.0,  2.0},   // Fixed Intermediate Waypoint A
        {35,  7.0,  8.0},   // Fixed Intermediate Waypoint B
        {49, 10.0, 10.0}    // End Position
    };

    int eq_row = 0;
    for (const auto& anchor : eq_anchors) {
        // X constraint: 1 * u[index] = x_val  =>  1 * u[index] - x_val = 0
        A_eq(eq_row, anchor.index) = 1.0;
        b_eq(eq_row) = -anchor.x_val;
        options.logger->info("Applying Equality Constraint: Waypoint {} FIXED at X = {}", anchor.index, anchor.x_val);
        eq_row++;

        // Y constraint: 1 * u[points_per_dim + index] = y_val
        A_eq(eq_row, points_per_dim + anchor.index) = 1.0;
        b_eq(eq_row) = -anchor.y_val;
        options.logger->info("Applying Equality Constraint: Waypoint {} FIXED at Y = {}", anchor.index, anchor.y_val);
        eq_row++;
    }
    problem.A = A_eq;
    problem.b = b_eq;

    // --- 5. Inequality Constraints Setup (Bounding Boxes) ---
    // Enforcing: Waypoint 10 (X <= 1.5, Y >= 3.0) and Waypoint 25 (X >= 6.0, Y <= 4.5)
    // Map to standard layout: C * u + d >= 0
    const int num_ineq_rows = 4;
    Eigen::MatrixXd C_ineq = Eigen::MatrixXd::Zero(num_ineq_rows, num_vars);
    Eigen::VectorXd d_ineq = Eigen::VectorXd::Zero(num_ineq_rows);

    // Row 0: Waypoint 10 Upper Bound X <= 1.5  => -1*x_10 + 1.5 >= 0
    C_ineq(0, 10) = -1.0;
    d_ineq(0) = 1.5;
    options.logger->info("Applying Inequality Constraint: Waypoint 10 must be BELOW X = 1.5");

    // Row 1: Waypoint 10 Lower Bound Y >= 3.0  => 1*y_10 - 3.0 >= 0
    C_ineq(1, points_per_dim + 10) = 1.0;
    d_ineq(1) = -3.0;
    options.logger->info("Applying Inequality Constraint: Waypoint 10 must be ABOVE Y = 3.0");

    // Row 2: Waypoint 25 Lower Bound X >= 6.0  => 1*x_25 - 6.0 >= 0
    C_ineq(2, 25) = 1.0;
    d_ineq(2) = -6.0;
    options.logger->info("Applying Inequality Constraint: Waypoint 25 must be ABOVE X = 6.0");

    // Row 3: Waypoint 25 Upper Bound Y <= 4.5  => -1*y_25 + 4.5 >= 0
    C_ineq(3, points_per_dim + 25) = -1.0;
    d_ineq(3) = 4.5;
    options.logger->info("Applying Inequality Constraint: Waypoint 25 must be BELOW Y = 4.5");

    problem.C = C_ineq;
    problem.d = d_ineq;

    // --- 6. Strictly Feasible Initial Guess Construction (u0) ---
    // A linear interpolation from (0,0) to (10,10) would violate the constraints at k=10 and k=25.
    // We explicitly route the initialization through safe intermediary nodes.
    Eigen::VectorXd u_init = Eigen::VectorXd::Zero(num_vars);
    
    // Intermediary routing nodes chosen to be strictly inside the constraint zones:
    // At k=10: x=1.0 (feebly under 1.5), y=3.5 (feebly over 3.0)
    // At k=25: x=6.5 (feebly over 6.0), y=4.0 (feebly under 4.5)
    for (int k = 0; k <= 10; ++k) {
        double t = static_cast<double>(k) / 10.0;
        u_init[k]                   = (1.0 - t) * 0.0 + t * 1.0; // X segment 1
        u_init[points_per_dim + k] = (1.0 - t) * 0.0 + t * 3.5; // Y segment 1
    }
    for (int k = 11; k <= 25; ++k) {
        double t = static_cast<double>(k - 10) / 15.0;
        u_init[k]                   = (1.0 - t) * 1.0 + t * 6.5; // X segment 2
        u_init[points_per_dim + k] = (1.0 - t) * 3.5 + t * 4.0; // Y segment 2
    }
    for (int k = 26; k <= N; ++k) {
        double t = static_cast<double>(k - 25) / 24.0;
        u_init[k]                   = (1.0 - t) * 6.5 + t * 10.0; // X segment 3
        u_init[points_per_dim + k] = (1.0 - t) * 4.0 + t * 10.0; // Y segment 3
    }
    problem.x0 = u_init;

    // Log Initialization Guard Verification
    Eigen::VectorXd feasibility_check = (C_ineq * u_init) + d_ineq;
    options.logger->info("Initial Guess Log Barrier Feasibility Clearance Values:");
    for(int r = 0; r < num_ineq_rows; ++r) {
        options.logger->info("  Inequality Row Constraint Margin [{}]: {}", r, feasibility_check[r]);
        if(feasibility_check[r] <= 0) {
            options.logger->error("FATAL: Initial guess violates inequality row {}", r);
        }
    }

    // 7. Initialize Solver Loop and Execute Run
    furiaopt::QPSolver solver(options, problem);
    furiaopt::Result result = solver.solve();

    // 8. Stream Out Final 2D Mapping Profile
    std::cout << "=====================================================================\n";
    std::cout << "          OPTIMIZED 2D SMOOTHED TRAJECTORY PROFILE (100 VARS)        \n";
    std::cout << "=====================================================================\n";
    std::cout << "Index\tX Coordinate\tY Coordinate\tActive Status Notes\n";
    std::cout << "---------------------------------------------------------------------\n";
    for (int i = 0; i < points_per_dim; ++i) {
        double x_res = result.x[i];
        double y_res = result.x[points_per_dim + i];
        
        std::cout << i << "\t" << x_res << "\t\t" << y_res;

        // Visual labels matching the output stream to your custom constraint nodes
        if (i == 0)  std::cout << "\t[EQ: Start Position Boundary Anchor]";
        if (i == 10) std::cout << "\t[INEQ active! Max X: 1.5, Min Y: 3.0]";
        if (i == 15) std::cout << "\t[EQ: Intermediary Anchor A]";
        if (i == 25) std::cout << "\t[INEQ active! Min X: 6.0, Max Y: 4.5]";
        if (i == 35) std::cout << "\t[EQ: Intermediary Anchor B]";
        if (i == N)  std::cout << "\t[EQ: End Position Boundary Anchor]";
        std::cout << "\n";
    }
    std::cout << "=====================================================================\n";
    std::cout << "Optimization Convergence Status: " << (result.summary.converged ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "Minimized Acceleration Profile Cost Matrix: " << result.summary.final_cost << "\n";
    std::cout << "=====================================================================\n";

    return 0;
}