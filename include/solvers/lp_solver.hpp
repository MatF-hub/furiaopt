#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
namespace furiaoptimizer{


class LPSolver{

    CostFunc cost_func_;

    std::reference_wrapper<const IPMSolverOptions> options_;
    std::reference_wrapper<const LPProblem> problem_;

public:

    //Constructor that takes the solver options and the problem structure
    LPSolver(const IPMSolverOptions& options, const LPProblem& problem);

    //Solve call
    Result solve();

    static Eigen::VectorXd computeFeasiblePoint(
    const Eigen::VectorXd& c,
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& d,
    const IPMSolverOptions& options);

private:
    void general_LP_solver(Result& result);
};

}