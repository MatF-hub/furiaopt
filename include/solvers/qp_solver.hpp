#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
namespace furiaopt{


class QPSolver{

    CostFunc cost_func_;
    Eigen::VectorXd x_0_;

    std::reference_wrapper<const IPMSolverOptions> options_;
    std::reference_wrapper<const QPProblem> problem_;
    std::shared_ptr<spdlog::logger> logger_;

public:

    //Constructor that takes the solver options and the problem structure
    QPSolver(const IPMSolverOptions& options, const QPProblem& problem);

    //Solve call
    Result solve();

private:
    void no_constraints_QP_solver(Result& result);
    void equality_constrained_QP_solver(Result& result);
    void general_QP_solver(Result& result);
};

}