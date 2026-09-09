#include "solvers/constrained_solver.hpp"
#include "config_loader.hpp"
#include "example_cli.hpp"
#include "utils.hpp"

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Delta robot trajectory-tracking SQP examples: joint-space decision variables, Cartesian
// tracking + smoothness cost via forward kinematics (never inverse kinematics), joint-limit box
// constraints, and circular keep-away obstacles. Two reference trajectories (pentagon, 8-pointed
// star), each solved with and without obstacles -- 4 problems total, same code path throughout.
// See docs/problems/delta_robot_opt_problem_math.md for the full derivation.

namespace {

constexpr double kPi = std::numbers::pi_v<double>;

// Geometry (mm) of delta robot
constexpr double R_B = 200.0;  // base radius
constexpr double R_E = 60.0;   // platform radius
constexpr double L1 = 250.0;   // bicep length
constexpr double L2 = 450.0;   // forearm length
constexpr double A_OFFSET = R_B - R_E;

constexpr double THETA_MIN = -25.0 * kPi / 180.0;
constexpr double THETA_MAX = 75.0 * kPi / 180.0;

// Z_HOME: 40mm below the max-reach singularity on the central axis. (experimentally found) 
// Z_PATH = Z_HOME - 250. Just a value in reach for the robot
// THETA_HOME: the (equal) joint angle of all 3 arms at HOME, from the same prototype.
constexpr double Z_HOME = -182.9324831207802;
constexpr double Z_PATH = -432.9324831207802;
constexpr double THETA_HOME = -0.20210866547421857;
constexpr double R_PATH = 150.0;

constexpr int SEGMENT_LEN = 9;  // 8 untracked interior points + 1 tracked endpoint, per segment

// Optimization problem weights
constexpr double W_TRACK = 1000.0;
constexpr double W_REG = 1.0;

std::array<double, 3> armPhi() {
    return {kPi / 2.0, kPi / 2.0 + 2.0 * kPi / 3.0, kPi / 2.0 + 4.0 * kPi / 3.0};
}

Eigen::Vector3d pointAtAngle(double radius, double angle_deg) {
    double angle = angle_deg * kPi / 180.0;
    return Eigen::Vector3d(radius * std::cos(angle), radius * std::sin(angle), Z_PATH);
}

// Pentagon vertices all lies on the minimum circle that inscribe the pentagon
std::array<Eigen::Vector3d, 5> pentagonVertices() {
    std::array<Eigen::Vector3d, 5> vertices;
    for (int i = 0; i < 5; ++i) vertices[i] = pointAtAngle(R_PATH, -90.0 + i * 72.0);
    return vertices;
}

// {8/3} star polygon: 8 points on a circle, visited in step-3 order (0,3,6,1,4,7,2,5) so a
// single continuous stroke covers every vertex before closing back to the first one.
std::array<Eigen::Vector3d, 8> starVertices() {
    std::array<Eigen::Vector3d, 8> vertices;
    for (int i = 0; i < 8; ++i) vertices[i] = pointAtAngle(R_PATH, -90.0 + i * 45.0);
    return vertices;
}
constexpr std::array<int, 8> star_traversal_order {0,3,6,1,4,7,2,5};

//Optimization variables are stacked on a column vector = [{\theta_1}_0...{\theta_1}_N,{\theta_2}_0...{\theta_2}_N,{\theta_3}_0...{\theta_3}_N ]^T
//Index in optimization vector of angle \theta_joint_number,k
int thetaIndex(int joint_number, int k, int n) { return joint_number * n + k; }

Eigen::Vector3d radialUnit(double phi) { return Eigen::Vector3d(std::cos(phi), std::sin(phi), 0.0); }

// c_j(theta_j): center of arm j's forearm sphere (elbow point offset by the platform radius).
Eigen::Vector3d elbowOffsetPoint(double theta, double phi) {
    Eigen::Vector3d u = radialUnit(phi);
    return (A_OFFSET + L1 * std::cos(theta)) * u + Eigen::Vector3d(0.0, 0.0, -L1 * std::sin(theta));
}

Eigen::Vector3d dElbowOffsetPoint(double theta, double phi) {
    Eigen::Vector3d u = radialUnit(phi);
    return -L1 * std::sin(theta) * u + Eigen::Vector3d(0.0, 0.0, -L1 * std::cos(theta));
}

// Forward kinematics via trilateration: p is the point at distance L2 from all 3 c_j(theta_j).
// Two solutions exist; the physical one is the lower-z root.
Eigen::Vector3d forwardKinematics(const Eigen::Vector3d& theta, const std::array<double, 3>& phi) {
    Eigen::Vector3d c0 = elbowOffsetPoint(theta(0), phi[0]);
    Eigen::Vector3d c1 = elbowOffsetPoint(theta(1), phi[1]);
    Eigen::Vector3d c2 = elbowOffsetPoint(theta(2), phi[2]);

    Eigen::Matrix<double, 2, 3> lhs;
    lhs.row(0) = 2.0 * (c0 - c1);
    lhs.row(1) = 2.0 * (c0 - c2);
    Eigen::Vector2d rhs(c0.squaredNorm() - c1.squaredNorm(), c0.squaredNorm() - c2.squaredNorm());
    Eigen::Vector3d p0 = lhs.completeOrthogonalDecomposition().solve(rhs);

    Eigen::Vector3d dir = (c0 - c1).cross(c0 - c2).normalized();
    Eigen::Vector3d diff = p0 - c0;
    double b = 2.0 * diff.dot(dir);
    double c = diff.squaredNorm() - L2 * L2;
    double disc = b * b - 4.0 * c;
    if (disc < 0.0) {
        throw std::runtime_error("forwardKinematics: theta is outside the reachable workspace (no trilateration solution)");
    }
    double t1 = (-b + std::sqrt(disc)) / 2.0;
    double t2 = (-b - std::sqrt(disc)) / 2.0;
    Eigen::Vector3d p1 = p0 + t1 * dir;
    Eigen::Vector3d p2 = p0 + t2 * dir;
    return p1.z() < p2.z() ? p1 : p2;
}

// dp/dtheta by implicit differentiation of F_j(p,theta) = |p-c_j(theta_j)|^2 - L2^2 = 0:
// dp/dtheta = -(dF/dp)^-1 (dF/dtheta). dF/dp has rows 2*(p-c_j); dF/dtheta is diagonal since
// F_j depends only on theta_j. A single 3x3 solve, no chain rule.
Eigen::Matrix3d fkJacobian(const Eigen::Vector3d& theta, const std::array<double, 3>& phi) {
    Eigen::Vector3d p = forwardKinematics(theta, phi);
    std::array<Eigen::Vector3d, 3> c, dc;
    for (int j = 0; j < 3; ++j) {
        c[j] = elbowOffsetPoint(theta(j), phi[j]);
        dc[j] = dElbowOffsetPoint(theta(j), phi[j]);
    }
    Eigen::Matrix3d dF_dp;
    Eigen::Vector3d dF_dtheta_diag;
    for (int j = 0; j < 3; ++j) {
        dF_dp.row(j) = 2.0 * (p - c[j]);
        dF_dtheta_diag(j) = -2.0 * (p - c[j]).dot(dc[j]);
    }
    return dF_dp.colPivHouseholderQr().solve(-dF_dtheta_diag.asDiagonal().toDenseMatrix());
}

// A circular (vertical-cylinder) keep-away obstacle in the horizontal plane.
struct Obstacle {
    std::string name;
    Eigen::Vector2d center;
    double radius;
};

Eigen::Vector2d midpointXY(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return 0.5 * (a.head<2>() + b.head<2>()); }
double chordHalfLength(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return (b.head<2>() - a.head<2>()).norm() / 2.0; }

// Joint limits: theta_min <= x <= theta_max, as h(x) = [x-theta_min; theta_max-x] >= 0.
Eigen::VectorXd jointLimitH(const Eigen::VectorXd& x, int num_vars) {
    Eigen::VectorXd h(2 * num_vars);
    h.head(num_vars) = x.array() - THETA_MIN;
    h.tail(num_vars) = THETA_MAX - x.array();
    return h;
}

Eigen::MatrixXd jointLimitGrad(int num_vars) {
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(num_vars, 2 * num_vars);
    grad.leftCols(num_vars) = Eigen::MatrixXd::Identity(num_vars, num_vars);
    grad.rightCols(num_vars) = -Eigen::MatrixXd::Identity(num_vars, num_vars);
    return grad;
}

// h_{o,k}(x) = |p_xy(theta_k) - c_o|^2 - r_o^2 >= 0, for every point k and every obstacle o --
// a keep-away, nonconvex feasible region (complement of a disk), applied over the whole
// trajectory (points far from an obstacle are simply slack).
Eigen::VectorXd obstacleH(const Eigen::VectorXd& x, const std::array<double, 3>& phi, const std::vector<Obstacle>& obstacles, int n) {
    Eigen::VectorXd h(n * obstacles.size());
    int row = 0;
    for (const auto& obs : obstacles) {
        for (int k = 0; k < n; ++k) {
            Eigen::Vector3d theta_k(x(thetaIndex(0, k, n)), x(thetaIndex(1, k, n)), x(thetaIndex(2, k, n)));
            Eigen::Vector2d p_xy = forwardKinematics(theta_k, phi).head<2>();
            h(row++) = (p_xy - obs.center).squaredNorm() - obs.radius * obs.radius;
        }
    }
    return h;
}

// dh/dtheta_k = 2*(p_xy-c_o)^T * [dp/dtheta]_{xy,:} -- chain rule through the same FK Jacobian,
// using only its x,y rows since the constraint doesn't depend on z.
Eigen::MatrixXd obstacleGrad(const Eigen::VectorXd& x, const std::array<double, 3>& phi, const std::vector<Obstacle>& obstacles, int n) {
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(3 * n, n * obstacles.size());
    int col = 0;
    for (const auto& obs : obstacles) {
        for (int k = 0; k < n; ++k) {
            Eigen::Vector3d theta_k(x(thetaIndex(0, k, n)), x(thetaIndex(1, k, n)), x(thetaIndex(2, k, n)));
            Eigen::Vector3d p = forwardKinematics(theta_k, phi);
            Eigen::Matrix3d J = fkJacobian(theta_k, phi);
            Eigen::RowVector3d dh_dtheta = 2.0 * (p.head<2>() - obs.center).transpose() * J.topRows<2>();
            for (int in = 0; in < 3; ++in) grad(thetaIndex(in, k, n), col) = dh_dtheta(in);
            ++col;
        }
    }
    return grad;
}

std::string referenceToString(const std::vector<Eigen::Vector3d>& targets) {
    std::ostringstream oss;
    oss.precision(9);
    for (std::size_t i = 0; i < targets.size(); ++i)
        oss << (i ? ";" : "") << targets[i].x() << "," << targets[i].y() << "," << targets[i].z();
    return oss.str();
}

std::string terminationReasonToString(furiaopt::TerminationReason reason) {
    switch (reason) {
        case furiaopt::TerminationReason::MaxIterations: return "MaxIterations";
        case furiaopt::TerminationReason::GradientTolerance: return "GradientTolerance";
        case furiaopt::TerminationReason::StepTolerance: return "StepTolerance";
        case furiaopt::TerminationReason::FunctionTolerance: return "FunctionTolerance";
        case furiaopt::TerminationReason::DirectSolve: return "DirectSolve";
    }
    return "Unknown";
}

// One trajectory-tracking problem: a reference sequence (home ... home) and an obstacle set.
struct ProblemConfig {
    std::string label;
    std::vector<Eigen::Vector3d> targets;  // includes home at both ends
    std::vector<Obstacle> obstacles;
};

}  // namespace

int main(int argc, char** argv) {
    auto args = furiaopt::parse_example_args(argc, argv);
    furiaopt::ConstrainedSolverOptions options =
        furiaopt::load_constrained_solver_options(args.config_dir + "/config.json", args.logs_output_dir);

    const std::array<double, 3> phi = armPhi();
    const Eigen::Vector3d home_point(0.0, 0.0, Z_HOME);
    const std::array<Eigen::Vector3d, 5> pentagon = pentagonVertices();
    const std::array<Eigen::Vector3d, 8> star = starVertices();

    options.logger->info("GEOMETRY R_B={} R_E={} L1={} L2={} theta_min_deg={} theta_max_deg={} w_track={} w_reg={}",
                          R_B, R_E, L1, L2, THETA_MIN * 180.0 / kPi, THETA_MAX * 180.0 / kPi, W_TRACK, W_REG);

    // Pentagon: home -> V1..V5 -> V1 (closing the loop) -> home.
    std::vector<Eigen::Vector3d> pentagon_targets = {home_point,   pentagon[0], pentagon[1], pentagon[2],
                                                      pentagon[3], pentagon[4], pentagon[0], home_point};
    const std::vector<Obstacle> pentagon_obstacles = {
        {"A", midpointXY(pentagon[1], pentagon[2]), 0.4 * chordHalfLength(pentagon[1], pentagon[2])},
        {"B", midpointXY(pentagon[3], pentagon[4]), 0.4 * chordHalfLength(pentagon[3], pentagon[4])},
    };

    // {8/3} star: home -> S0,S3,S6,S1,S4,S7,S2,S5 (single stroke) -> S0 (closing) -> home.
    std::vector<Eigen::Vector3d> star_targets = {home_point};
    for (int i : star_traversal_order) star_targets.push_back(star[i]);
    star_targets.push_back(star[star_traversal_order[0]]);
    star_targets.push_back(home_point);
    // Obstacles on 2 opposite star edges (180 deg apart by symmetry) -- smaller fraction than
    // the pentagon's since star edges are long diagonals crossing near the center. Named by
    // traversal-order position (edge 0: S0->S3, edge 4: S4->S7) rather than indexing into
    // star_targets, so they don't silently shift if the traversal/lead-in sequence changes.
    const Eigen::Vector3d& star_edge0_start = star[star_traversal_order[0]];
    const Eigen::Vector3d& star_edge0_end = star[star_traversal_order[1]];
    const Eigen::Vector3d& star_edge4_start = star[star_traversal_order[4]];
    const Eigen::Vector3d& star_edge4_end = star[star_traversal_order[5]];
    const std::vector<Obstacle> star_obstacles = {
        {"A", midpointXY(star_edge0_start, star_edge0_end), 0.3 * chordHalfLength(star_edge0_start, star_edge0_end)},
        {"B", midpointXY(star_edge4_start, star_edge4_end), 0.3 * chordHalfLength(star_edge4_start, star_edge4_end)},
    };

    const std::vector<ProblemConfig> configs = {
        {"Problem 1 (pentagon, no obstacles)", pentagon_targets, {}},
        {"Problem 2 (pentagon, two obstacles)", pentagon_targets, pentagon_obstacles},
        {"Problem 3 (star, no obstacles)", star_targets, {}},
        {"Problem 4 (star, two obstacles)", star_targets, star_obstacles},
    };

    for (const auto& cfg : configs) {
        const int n = static_cast<int>(cfg.targets.size() - 1) * SEGMENT_LEN + 1;
        const int num_vars = 3 * n;
        std::vector<int> tracked_k;
        for (int k = 0; k < n; k += SEGMENT_LEN) tracked_k.push_back(k);

        options.logger->info("=== METHOD {} ===", cfg.label);
        options.logger->info("REFERENCE points={}", referenceToString(cfg.targets));
        for (const auto& obs : cfg.obstacles)
            options.logger->info("OBSTACLE name={} center_x={} center_y={} radius={}", obs.name, obs.center.x(),
                                  obs.center.y(), obs.radius);

        furiaopt::LSProblem problem;
        problem.x0 = Eigen::VectorXd::Constant(num_vars, THETA_HOME); //Initialize the robot to be at home for the whole time.

        const auto& targets = cfg.targets;
        problem.residual_func = [phi, tracked_k, targets, n](const Eigen::VectorXd& x) {
            std::vector<Eigen::Vector3d> p(n);
            for (int k = 0; k < n; ++k) {
                Eigen::Vector3d theta_k(x(thetaIndex(0, k, n)), x(thetaIndex(1, k, n)), x(thetaIndex(2, k, n)));
                p[k] = forwardKinematics(theta_k, phi);
            }

            Eigen::VectorXd r(3 * tracked_k.size() + 3 * (n - 1));
            int row = 0;
            for (std::size_t i = 0; i < tracked_k.size(); ++i) {
                r.segment(row, 3) = std::sqrt(W_TRACK) * (p[tracked_k[i]] - targets[i]);
                row += 3;
            }
            for (int k = 0; k < n - 1; ++k) {
                r.segment(row, 3) = std::sqrt(W_REG) * (p[k + 1] - p[k]);
                row += 3;
            }
            return r;
        };

        problem.gradient_residual_func = [phi, tracked_k, n](const Eigen::VectorXd& x) {
            std::vector<Eigen::Vector3d> theta(n);
            std::vector<Eigen::Matrix3d> J(n);
            for (int k = 0; k < n; ++k) {
                theta[k] = Eigen::Vector3d(x(thetaIndex(0, k, n)), x(thetaIndex(1, k, n)), x(thetaIndex(2, k, n)));
                J[k] = fkJacobian(theta[k], phi);
            }

            int num_vars = 3 * n;
            int m = 3 * tracked_k.size() + 3 * (n - 1);
            Eigen::MatrixXd Jr = Eigen::MatrixXd::Zero(num_vars, m);

            auto addBlock = [&](int k, int row, double weight, double sign) {
                for (int out = 0; out < 3; ++out)
                    for (int in = 0; in < 3; ++in)
                        Jr(thetaIndex(in, k, n), row + out) += sign * weight * J[k](out, in);
            };

            int row = 0;
            for (std::size_t i = 0; i < tracked_k.size(); ++i) {
                addBlock(tracked_k[i], row, std::sqrt(W_TRACK), 1.0);
                row += 3;
            }
            for (int k = 0; k < n - 1; ++k) {
                addBlock(k + 1, row, std::sqrt(W_REG), 1.0);
                addBlock(k, row, std::sqrt(W_REG), -1.0);
                row += 3;
            }
            return Jr;
        };

        problem.inequality_constraint_func = [phi, obstacles = cfg.obstacles, n, num_vars](const Eigen::VectorXd& x) {
            if (obstacles.empty()) return jointLimitH(x, num_vars);
            Eigen::VectorXd h_obs = obstacleH(x, phi, obstacles, n);
            Eigen::VectorXd h(2 * num_vars + h_obs.size());
            h << jointLimitH(x, num_vars), h_obs;
            return h;
        };
        problem.gradient_inequality_constraint_func = [phi, obstacles = cfg.obstacles, n, num_vars](const Eigen::VectorXd& x) {
            if (obstacles.empty()) return jointLimitGrad(num_vars);
            Eigen::MatrixXd g_obs = obstacleGrad(x, phi, obstacles, n);
            Eigen::MatrixXd g(num_vars, 2 * num_vars + g_obs.cols());
            g << jointLimitGrad(num_vars), g_obs;
            return g;
        };

        furiaopt::ConstrainedSolver solver(options, problem);
        auto t0 = std::chrono::steady_clock::now();
        furiaopt::Result result = solver.solve();
        double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        options.logger->info(
            "SOLVE elapsed_ms={:.4f} iterations={} final_cost={:.6f} converged={} termination_reason={}", elapsed_ms,
            result.summary.iterations, result.summary.final_cost, result.summary.converged,
            terminationReasonToString(result.summary.termination_reason));
        options.logger->info("RESULT x={}", furiaopt::utils::vec_to_string(result.x));
        
        //Debug cout statement per-problem
        std::cout << cfg.label << ": converged=" << (result.summary.converged ? "yes" : "no")
                   << " termination_reason=" << terminationReasonToString(result.summary.termination_reason)
                   << " iterations=" << result.summary.iterations << " final_cost=" << result.summary.final_cost
                   << " elapsed_ms=" << elapsed_ms << "\n";

        for (std::size_t i = 0; i < tracked_k.size(); ++i) {
            int k = tracked_k[i];
            Eigen::Vector3d theta_k(result.x(thetaIndex(0, k, n)), result.x(thetaIndex(1, k, n)), result.x(thetaIndex(2, k, n)));
            Eigen::Vector3d p = forwardKinematics(theta_k, phi);
            std::cout << "  k=" << k << "\ttarget=(" << targets[i].transpose() << ")\tsolved=(" << p.transpose()
                       << ")\terr=" << (p - targets[i]).norm() << "mm\n";
        }

        std::cout << "  theta range: [" << result.x.minCoeff() * 180.0 / kPi << ", "
                   << result.x.maxCoeff() * 180.0 / kPi << "] deg (limits [" << THETA_MIN * 180.0 / kPi << ", "
                   << THETA_MAX * 180.0 / kPi << "])\n";
    }

    return 0;
}
