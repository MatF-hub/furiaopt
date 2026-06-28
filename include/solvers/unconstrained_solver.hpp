#pragma once

#include <Eigen/Dense>
#include "solver_config.hpp"
#include "types.hpp"
#include "direction_strategy.hpp"
#include <variant>
namespace furiaopt{

class UnconstrainedSolver{

private:
    //The same solver can be used for both NLP and LS problems.
    //At this scope use a type-erasure approach to decouple the solver from the specific problem type.
    CostFunc cost_func_;
    GradientFunc gradient_func_;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&)> get_direction_func_;

    Eigen::VectorXd x0_;

    std::reference_wrapper<const UnconstrainedSolverOptions> options_;
    std::shared_ptr<spdlog::logger> logger_;

public:

    //Constructor that takes the solver options and the problem structure
    UnconstrainedSolver(const UnconstrainedSolverOptions& options, const NLPProblem& problem);
    UnconstrainedSolver(const UnconstrainedSolverOptions& options, const LSProblem& problem);


    //Solve call
    Result solve();
};

}
