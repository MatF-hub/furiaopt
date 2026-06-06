#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
#include "direction_strategy.hpp"
namespace furiaoptimizer{


class ConstrainedSolver{

    SolverOptions& options_;
    NLPProblem& problem_;
    std::unique_ptr<DirectionStrategy> direction_strategy_;

public:

    //Constructor that takes the solver options and the problem structure
    ConstrainedSolver(const SolverOptions& options, const NLPProblem& problem);

    //Solve call
    Result solve();

private:
    //Sequential quadratic programming solver for non-linear problems
    void SQP_solver(Result& result, const NLPProblem& problem_);
};

}
