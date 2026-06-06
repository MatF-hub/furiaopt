#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
#include "direction_strategy.hpp"
namespace furiaoptimizer{


class ConstrainedSolver{

    SolverOptions options_;
    Problem problem_;
    std::unique_ptr<DirectionStrategy> direction_strategy_;

public:

    //Constructor that takes the solver options and the problem structure
    ConstrainedSolver(const SolverOptions& options, const Problem& problem);

    //Solve call
    Result solve();

private:
    //LP solver for linear problems, it directly solves the linear program using an interior point method
    void LP_solver(Result& result, const Problem& problem_);
    //QP solver for quadratic problems with only linear constraints, it directly solves the KKT conditions
    void QP_solver(Result& result, const Problem& problem_);
    //Sequential quadratic programming solver for non-linear problems
    void SQP_solver(Result& result, const Problem& problem_);
};

}
