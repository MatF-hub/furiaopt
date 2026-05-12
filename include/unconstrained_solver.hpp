#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
#include "direction_strategy.hpp"
namespace furiaoptimizer{


class Solver{

    SolverOptions options_;
    Problem problem_;
    std::unique_ptr<DirectionStrategy> direction_strategy_;

public:

    //Constructor that takes the solver options and the problem structure
    Solver(const SolverOptions& options, const Problem& problem);

    //Solve call
    Result solve();
    void non_linear_solver(Result& result, const Problem& problem_);
};

}
